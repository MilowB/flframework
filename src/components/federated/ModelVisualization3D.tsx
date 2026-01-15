import { useState, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere } from '@react-three/drei';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { RoundMetrics } from '@/lib/federated/types';
import { Model3DPosition, RoundSnapshot3D, computeModelPositions, flattenModelWeights } from '@/lib/federated/visualization/pca';
import * as THREE from 'three';

interface ModelVisualization3DProps {
  history: RoundMetrics[];
  clientModels?: Map<string, { weights: { layers: number[][]; bias: number[] }; name: string }>;
  clusterModels?: Map<string, { layers: number[][]; bias: number[] }>;
  globalModel?: { layers: number[][]; bias: number[]; version: number } | null;
}

// Individual model sphere component
function ModelSphere({ 
  position, 
  color, 
  name, 
  type, 
  isHovered, 
  onHover 
}: { 
  position: [number, number, number];
  color: string;
  name: string;
  type: 'client' | 'cluster' | 'global';
  isHovered: boolean;
  onHover: (hovered: boolean) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Animate scale on hover
  useFrame(() => {
    if (meshRef.current) {
      const targetScale = isHovered ? 1.3 : 1;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  const radius = type === 'global' ? 0.4 : type === 'cluster' ? 0.3 : 0.2;
  
  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => onHover(true)}
        onPointerOut={() => onHover(false)}
      >
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color}
          emissiveIntensity={isHovered ? 0.5 : 0.2}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
      {isHovered && (
        <Text
          position={[0, radius + 0.3, 0]}
          fontSize={0.25}
          color="white"
          anchorX="center"
          anchorY="bottom"
        >
          {name}
        </Text>
      )}
    </group>
  );
}

// Scene component
function Scene({ positions }: { positions: Model3DPosition[] }) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      
      {/* Grid helper */}
      <gridHelper args={[20, 20, '#333', '#222']} position={[0, -3, 0]} />
      
      {/* Axis lines */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-6, 0, 0, 6, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ff4444" opacity={0.5} transparent />
      </line>
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, -6, 0, 0, 6, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#44ff44" opacity={0.5} transparent />
      </line>
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, -6, 0, 0, 6])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#4444ff" opacity={0.5} transparent />
      </line>
      
      {/* Model spheres */}
      {positions.map((model) => (
        <ModelSphere
          key={model.id}
          position={model.position}
          color={model.color}
          name={model.name}
          type={model.type}
          isHovered={hoveredId === model.id}
          onHover={(hovered) => setHoveredId(hovered ? model.id : null)}
        />
      ))}
      
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={3}
        maxDistance={30}
      />
    </>
  );
}

export function ModelVisualization3D({ 
  history, 
  clientModels, 
  clusterModels, 
  globalModel 
}: ModelVisualization3DProps) {
  const [currentRound, setCurrentRound] = useState(0);
  
  // Compute positions for current round
  const positions = useMemo(() => {
    if (!history.length) return [];
    
    const roundIndex = Math.min(currentRound, history.length - 1);
    const roundData = history[roundIndex];
    
    // Get client metrics from the round
    const clientList: { id: string; name: string; weights: { layers: number[][]; bias: number[]; version: number } }[] = [];
    
    // Use clientModels if available, otherwise create placeholder
    if (clientModels) {
      clientModels.forEach((model, id) => {
        clientList.push({
          id,
          name: model.name,
          weights: { ...model.weights, version: 0 }
        });
      });
    } else if (roundData.clientMetrics) {
      // Create placeholder positions based on client metrics
      roundData.clientMetrics.forEach((cm, idx) => {
        clientList.push({
          id: cm.clientId,
          name: cm.clientName,
          weights: {
            layers: [[idx * 0.1, idx * 0.2]],
            bias: [idx * 0.05],
            version: 0
          }
        });
      });
    }
    
    // Get cluster models
    const clusterList: { id: string; weights: { layers: number[][]; bias: number[]; version: number } }[] = [];
    if (clusterModels) {
      clusterModels.forEach((model, id) => {
        clusterList.push({
          id,
          weights: { ...model, version: 0 }
        });
      });
    } else if (roundData.clusterMetrics) {
      roundData.clusterMetrics.forEach((cm, idx) => {
        clusterList.push({
          id: `cluster-${cm.clusterId}`,
          weights: {
            layers: [[idx * 0.3, idx * 0.4]],
            bias: [idx * 0.1],
            version: 0
          }
        });
      });
    }
    
    // Use globalModel if available
    const global = globalModel || null;
    
    return computeModelPositions(clientList, clusterList, global, 'aggregated');
  }, [history, currentRound, clientModels, clusterModels, globalModel]);
  
  const maxRound = Math.max(0, history.length - 1);
  
  const handlePrevRound = () => {
    setCurrentRound(prev => Math.max(0, prev - 1));
  };
  
  const handleNextRound = () => {
    setCurrentRound(prev => Math.min(maxRound, prev + 1));
  };
  
  if (history.length === 0) {
    return (
      <div className="h-[400px] flex items-center justify-center text-muted-foreground">
        <p>Lancez l'entraînement pour voir la visualisation 3D</p>
      </div>
    );
  }
  
  return (
    <div className="flex flex-col gap-4">
      {/* Legend */}
      <div className="flex flex-wrap gap-4 justify-center text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-white" />
          <span className="text-muted-foreground">Modèle Global</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-violet-500" />
          <span className="text-muted-foreground">Modèles Clusters</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-500" />
          <span className="text-muted-foreground">Modèles Clients</span>
        </div>
      </div>
      
      {/* 3D Canvas */}
      <div className="h-[400px] w-full rounded-lg overflow-hidden bg-black/50 border border-border">
        <Canvas
          camera={{ position: [8, 6, 8], fov: 50 }}
          style={{ background: 'transparent' }}
        >
          <Scene positions={positions} />
        </Canvas>
      </div>
      
      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <Button 
          variant="outline" 
          size="icon"
          onClick={handlePrevRound}
          disabled={currentRound === 0}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        
        <div className="px-4 py-2 rounded-lg bg-muted/30 border border-border min-w-[120px] text-center">
          <span className="text-lg font-mono font-bold">
            Round {currentRound + 1}
          </span>
          <span className="text-muted-foreground text-sm ml-1">
            / {maxRound + 1}
          </span>
        </div>
        
        <Button 
          variant="outline" 
          size="icon"
          onClick={handleNextRound}
          disabled={currentRound >= maxRound}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      
      {/* Phase indicator */}
      <div className="flex justify-center gap-2 text-xs text-muted-foreground">
        <span>Utilisez la souris pour zoomer et faire pivoter</span>
      </div>
    </div>
  );
}
