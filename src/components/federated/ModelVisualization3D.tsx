import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Sphere } from '@react-three/drei';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Download } from 'lucide-react';
import { RoundMetrics } from '@/lib/federated/types';
import { Model3DPosition, RoundSnapshot3D, flattenModelWeights } from '@/lib/federated/visualization/pca';
import * as THREE from 'three';
import GIF from 'gif.js';
import { toast } from 'sonner';

// Import computePCAProjection and CLUSTER_COLORS from pca module
function computePCAProjection(vectors: number[][], numComponents: number = 3): number[][] {
  if (vectors.length === 0) return [];
  
  const dim = vectors[0].length;
  const n = vectors.length;
  
  const projectionVectors: number[][] = [];
  
  for (let comp = 0; comp < numComponents; comp++) {
    let v = new Array(dim).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    v = v.map(x => x / (norm || 1));
    
    for (let iter = 0; iter < 10; iter++) {
      const scores = vectors.map(row => row.reduce((sum, x, i) => sum + x * v[i], 0));
      const newV = new Array(dim).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < dim; j++) {
          newV[j] += vectors[i][j] * scores[i];
        }
      }
      
      for (const prev of projectionVectors) {
        const dot = newV.reduce((sum, x, i) => sum + x * prev[i], 0);
        for (let i = 0; i < dim; i++) {
          newV[i] -= dot * prev[i];
        }
      }
      
      norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
      v = newV.map(x => x / (norm || 1));
    }
    
    projectionVectors.push(v);
  }
  
  return projectionVectors;
}

const CLUSTER_COLORS = [
  '#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6',
  '#ec4899', '#eab308', '#a855f7', '#14b8a6', '#84cc16', '#fb923c',
];

const GLOBAL_COLOR = '#ffffff';

interface ModelVisualization3DProps {
  history: RoundMetrics[];
  clientModels?: Map<string, { weights: { layers: number[][]; bias: number[] }; name: string }>;
  clusterModels?: Map<string, { layers: number[][]; bias: number[] }>;
  globalModel?: { layers: number[][]; bias: number[]; version: number } | null;
  loadedVisualizations?: { round: number; models: Model3DPosition[] }[];
}

// Individual model sphere/box component
function ModelSphere({ 
  position, 
  color, 
  name, 
  type, 
  isHovered, 
  onHover,
  shape = 'sphere'
}: { 
  position: [number, number, number];
  color: string;
  name: string;
  type: 'client' | 'cluster' | 'global';
  isHovered: boolean;
  onHover: (hovered: boolean) => void;
  shape?: 'sphere' | 'box';
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const targetPosition = useRef(new THREE.Vector3(...position));
  const isInitialized = useRef(false);
  const [showBox, setShowBox] = useState(shape !== 'box'); // Boxes (clusters) start hidden
  const [currentColor, setCurrentColor] = useState(color); // Track current color for delayed update
  
  // Initialize position on first render
  useEffect(() => {
    if (!isInitialized.current && groupRef.current) {
      groupRef.current.position.set(...position);
      isInitialized.current = true;
    }
  }, []);
  
  // Update target position when prop changes
  useEffect(() => {
    targetPosition.current.set(...position);
  }, [position]);
  
  // Update color with delay for boxes (clusters)
  useEffect(() => {
    if (shape === 'box') {
      const timer = setTimeout(() => {
        setCurrentColor(color);
      }, 500);
      return () => clearTimeout(timer);
    } else {
      // Immediate color change for spheres
      setCurrentColor(color);
    }
  }, [color, shape]);
  
  // Animate position and scale
  useFrame(() => {
    if (groupRef.current && shape !== 'box') {
      // Smooth position transition only for spheres (clients), not boxes (clusters)
      groupRef.current.position.lerp(targetPosition.current, 0.1);
    }
    
    if (meshRef.current) {
      // Animate scale on hover
      const targetScale = isHovered ? 1.3 : 1;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });
  
  // Update group position with delay for boxes (clusters)
  useEffect(() => {
    if (shape === 'box' && groupRef.current) {
      // Hide box first
      setShowBox(false);
      
      // Wait 500ms for client transitions to complete, then show and position box
      const timer = setTimeout(() => {
        if (groupRef.current) {
          groupRef.current.position.set(...position);
          setShowBox(true);
        }
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [position, shape]);
  
  const radius = type === 'global' ? 0.4 : type === 'cluster' ? 0.3 : 0.2;
  const boxSize = type === 'global' ? 0.7 : type === 'cluster' ? 0.5 : 0.35;
  
  return (
    <group ref={groupRef}>
      {showBox && (
        <mesh
          ref={meshRef}
          onPointerOver={() => onHover(true)}
          onPointerOut={() => onHover(false)}
        >
          {shape === 'box' ? (
            <boxGeometry args={[boxSize, boxSize, boxSize]} />
          ) : (
            <sphereGeometry args={[radius, 32, 32]} />
          )}
          <meshStandardMaterial 
            color={currentColor} 
            emissive={currentColor}
            emissiveIntensity={isHovered ? 0.5 : 0.2}
            roughness={0.3}
            metalness={0.7}
          />
        </mesh>
      )}
      {/* Always show label for clients, hover for others */}
      {showBox && (type === 'client' || isHovered) && (
        <Text
          position={[0, (shape === 'box' ? boxSize/2 : radius) + 0.3, 0]}
          fontSize={type === 'client' ? 0.2 : 0.25}
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

// Helper component to capture canvas reference
function CanvasCapture({ canvasRef }: { canvasRef: React.MutableRefObject<HTMLCanvasElement | null> }) {
  const { gl } = useThree();
  
  useEffect(() => {
    canvasRef.current = gl.domElement;
  }, [gl, canvasRef]);
  
  return null;
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
      
      {/* Model spheres and boxes */}
      {positions.map((model) => (
        <ModelSphere
          key={model.id}
          position={model.position}
          color={model.color}
          name={model.name}
          type={model.type}
          isHovered={hoveredId === model.id}
          onHover={(hovered) => setHoveredId(hovered ? model.id : null)}
          shape={model.type === 'cluster' ? 'box' : 'sphere'}
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

// Export function to compute and extract all 3D positions for saving
export function compute3DPositions(history: RoundMetrics[]): { round: number; models: Model3DPosition[] }[] {
  if (!history.length) return [];
  
  // Build a stable cluster color mapping across all rounds
  const clusterSignatureToColorIndex = new Map<string, number>();
  let nextColorIndex = 0;
  
  const getClusterColorIndex = (clientIds: string[]): number => {
    const signature = [...clientIds].sort().join(',');
    if (!clusterSignatureToColorIndex.has(signature)) {
      clusterSignatureToColorIndex.set(signature, nextColorIndex);
      nextColorIndex = (nextColorIndex + 1) % CLUSTER_COLORS.length;
    }
    return clusterSignatureToColorIndex.get(signature)!;
  };
  
  // Collect all models from all rounds
  const allClientVectors: number[][] = [];
  const allClientMeta: { roundIndex: number; id: string; name: string; colorIndex?: number }[] = [];
  const allClusterVectors: number[][] = [];
  const allClusterMeta: { roundIndex: number; id: string; name: string; colorIndex: number }[] = [];
  const allGlobalVectors: number[][] = [];
  const allGlobalMeta: { roundIndex: number; id: string; name: string }[] = [];
  
  history.forEach((roundData, roundIndex) => {
    const clientToColorIndex = new Map<string, number>();
    if (roundData.clusterMetrics) {
      roundData.clusterMetrics.forEach((cm) => {
        const colorIndex = getClusterColorIndex(cm.clientIds);
        cm.clientIds.forEach(clientId => {
          clientToColorIndex.set(clientId, colorIndex);
        });
      });
    }
    
    if (roundData.clientMetrics) {
      roundData.clientMetrics.forEach((cm) => {
        if (cm.weights) {
          const colorIndex = clientToColorIndex.get(cm.clientId);
          allClientVectors.push(flattenModelWeights(cm.weights));
          allClientMeta.push({
            roundIndex,
            id: cm.clientId,
            name: cm.clientName,
            colorIndex
          });
        }
      });
    }
    
    if (roundData.clusterMetrics) {
      roundData.clusterMetrics.forEach((cm) => {
        if (cm.weights) {
          const colorIndex = getClusterColorIndex(cm.clientIds);
          allClusterVectors.push(flattenModelWeights(cm.weights));
          allClusterMeta.push({
            roundIndex,
            id: `cluster-${cm.clusterId}`,
            name: `Cluster ${cm.clusterId}`,
            colorIndex
          });
        }
      });
    }
    
    if (roundData.globalModelWeights) {
      allGlobalVectors.push(flattenModelWeights(roundData.globalModelWeights));
      allGlobalMeta.push({
        roundIndex,
        id: 'global',
        name: 'Global'
      });
    }
  });
  
  if (allClientVectors.length === 0) return [];
  
  // Compute PCA on all models together
  const allVectors = [...allClientVectors, ...allClusterVectors, ...allGlobalVectors];
  const allMeta = [...allClientMeta, ...allClusterMeta, ...allGlobalMeta];
  
  const dim = allVectors[0].length;
  const mean = new Array(dim).fill(0);
  for (const vec of allVectors) {
    for (let i = 0; i < dim; i++) {
      mean[i] += vec[i] / allVectors.length;
    }
  }
  
  const projVectors = computePCAProjection(allVectors.map(vec => vec.map((v, i) => v - mean[i])), 3);
  
  if (projVectors.length < 3) return [];
  
  // Project all vectors
  const positions = allVectors.map((vec, i) => {
    const centered = vec.map((v, j) => v - mean[j]);
    const coords: [number, number, number] = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      coords[d] = centered.reduce((sum, v, j) => sum + v * projVectors[d][j], 0);
    }
    
    const isGlobal = i >= allClientMeta.length + allClusterMeta.length;
    const isCluster = !isGlobal && i >= allClientMeta.length;
    const meta = allMeta[i];
    
    // Get colorIndex from client or cluster meta arrays directly
    let colorIndex: number | undefined;
    if (!isGlobal && !isCluster) {
      // It's a client - find in allClientMeta
      const clientIdx = i;
      colorIndex = allClientMeta[clientIdx]?.colorIndex;
    } else if (isCluster) {
      // It's a cluster - find in allClusterMeta
      const clusterIdx = i - allClientMeta.length;
      colorIndex = allClusterMeta[clusterIdx]?.colorIndex;
    }
    
    return {
      roundIndex: meta.roundIndex,
      id: meta.id,
      name: isGlobal ? meta.name : isCluster ? meta.name : meta.id.replace('client-', ''),
      type: (isGlobal ? 'global' : isCluster ? 'cluster' : 'client') as 'client' | 'cluster' | 'global',
      position: coords,
      clusterIdx: colorIndex,
      color: isGlobal 
        ? GLOBAL_COLOR 
        : colorIndex !== undefined 
          ? CLUSTER_COLORS[colorIndex % CLUSTER_COLORS.length] 
          : CLUSTER_COLORS[0]
    };
  });
  
  // Normalize all positions to same scale
  const allCoords = positions.flatMap(p => p.position);
  const maxAbs = Math.max(...allCoords.map(Math.abs), 1);
  const scale = 5 / maxAbs;
  
  const normalizedPositions = positions.map(p => ({
    ...p,
    position: p.position.map(c => c * scale) as [number, number, number]
  }));
  
  // Group by round
  const byRound = new Map<number, Model3DPosition[]>();
  normalizedPositions.forEach(pos => {
    if (!byRound.has(pos.roundIndex)) {
      byRound.set(pos.roundIndex, []);
    }
    byRound.get(pos.roundIndex)!.push(pos);
  });
  
  return Array.from(byRound.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([round, models]) => ({ round, models }));
}

export function ModelVisualization3D({ 
  history, 
  clientModels, 
  clusterModels, 
  globalModel,
  loadedVisualizations
}: ModelVisualization3DProps) {
  const [currentRound, setCurrentRound] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Compute PCA once for all rounds to have a consistent reference frame
  const allRoundsPositions = useMemo(() => {
    // If we have pre-loaded visualizations, use them instead of computing
    if (loadedVisualizations && loadedVisualizations.length > 0) {
      console.log('=== Using pre-loaded 3D visualizations ===');
      return loadedVisualizations.flatMap(({ models }) => models);
    }
    
    if (!history.length) return [];
    
    console.log('=== Computing PCA for all rounds ===');
    
    // Build a stable cluster color mapping across all rounds
    // based on cluster composition (sorted client IDs)
    const clusterSignatureToColorIndex = new Map<string, number>();
    let nextColorIndex = 0;
    
    const getClusterColorIndex = (clientIds: string[]): number => {
      const signature = [...clientIds].sort().join(',');
      if (!clusterSignatureToColorIndex.has(signature)) {
        clusterSignatureToColorIndex.set(signature, nextColorIndex);
        nextColorIndex = (nextColorIndex + 1) % CLUSTER_COLORS.length;
      }
      return clusterSignatureToColorIndex.get(signature)!;
    };
    
    // Collect all models from all rounds
    const allClientVectors: number[][] = [];
    const allClientMeta: { roundIndex: number; id: string; name: string; colorIndex?: number }[] = [];
    const allClusterVectors: number[][] = [];
    const allClusterMeta: { roundIndex: number; id: string; name: string; colorIndex: number }[] = [];
    const allGlobalVectors: number[][] = [];
    const allGlobalMeta: { roundIndex: number; id: string; name: string }[] = [];
    
    history.forEach((roundData, roundIndex) => {
      // Build cluster color mapping for this round
      const clientToColorIndex = new Map<string, number>();
      if (roundData.clusterMetrics) {
        roundData.clusterMetrics.forEach((cm) => {
          const colorIndex = getClusterColorIndex(cm.clientIds);
          cm.clientIds.forEach(clientId => {
            clientToColorIndex.set(clientId, colorIndex);
          });
        });
      }
      
      // Collect client models for this round
      if (roundData.clientMetrics) {
        roundData.clientMetrics.forEach((cm) => {
          if (cm.weights) {
            const colorIndex = clientToColorIndex.get(cm.clientId);
            
            allClientVectors.push(flattenModelWeights(cm.weights));
            allClientMeta.push({
              roundIndex,
              id: cm.clientId,
              name: cm.clientName,
              colorIndex
            });
          }
        });
      }
      
      // Collect cluster models for this round
      if (roundData.clusterMetrics) {
        roundData.clusterMetrics.forEach((cm) => {
          if (cm.weights) {
            const colorIndex = getClusterColorIndex(cm.clientIds);
            allClusterVectors.push(flattenModelWeights(cm.weights));
            allClusterMeta.push({
              roundIndex,
              id: `cluster-${cm.clusterId}`,
              name: `Cluster ${cm.clusterId}`,
              colorIndex
            });
          }
        });
      }
      
      // Collect global model for this round
      if (roundData.globalModelWeights) {
        allGlobalVectors.push(flattenModelWeights(roundData.globalModelWeights));
        allGlobalMeta.push({
          roundIndex,
          id: 'global',
          name: 'Global'
        });
      }
    });
    
    if (allClientVectors.length === 0) {
      console.log('No models with weights found');
      return [];
    }
    
    console.log(`Computing PCA with ${allClientVectors.length} clients + ${allClusterVectors.length} clusters + ${allGlobalVectors.length} global models`);
    console.log(`Found ${clusterSignatureToColorIndex.size} unique cluster compositions`);
    
    // Compute PCA on all models together
    const allVectors = [...allClientVectors, ...allClusterVectors, ...allGlobalVectors];
    const allMeta = [...allClientMeta, ...allClusterMeta, ...allGlobalMeta];
    
    const dim = allVectors[0].length;
    const mean = new Array(dim).fill(0);
    for (const vec of allVectors) {
      for (let i = 0; i < dim; i++) {
        mean[i] += vec[i] / allVectors.length;
      }
    }
    
    // Compute PCA projection
    const projVectors = computePCAProjection(allVectors.map(vec => vec.map((v, i) => v - mean[i])), 3);
    
    if (projVectors.length < 3) {
      console.log('PCA failed, not enough components');
      return [];
    }
    
    // Project all vectors
    const positions = allVectors.map((vec, i) => {
      const centered = vec.map((v, j) => v - mean[j]);
      const coords: [number, number, number] = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        coords[d] = centered.reduce((sum, v, j) => sum + v * projVectors[d][j], 0);
      }
      
      const isGlobal = i >= allClientMeta.length + allClusterMeta.length;
      const isCluster = !isGlobal && i >= allClientMeta.length;
      const meta = allMeta[i];
      
      // Get colorIndex from the appropriate meta array
      let colorIndex: number | undefined;
      if (!isGlobal && !isCluster && i < allClientMeta.length) {
        colorIndex = allClientMeta[i].colorIndex;
      } else if (isCluster) {
        colorIndex = allClusterMeta[i - allClientMeta.length].colorIndex;
      }
      
      return {
        roundIndex: meta.roundIndex,
        id: meta.id,
        name: isGlobal ? meta.name : isCluster ? meta.name : meta.id.replace('client-', ''),
        type: (isGlobal ? 'global' : isCluster ? 'cluster' : 'client') as 'client' | 'cluster' | 'global',
        position: coords,
        clusterIdx: colorIndex,
        color: isGlobal 
          ? GLOBAL_COLOR 
          : colorIndex !== undefined 
            ? CLUSTER_COLORS[colorIndex % CLUSTER_COLORS.length] 
            : CLUSTER_COLORS[0]
      };
    });
    
    // Normalize all positions to same scale
    const allCoords = positions.flatMap(p => p.position);
    const maxAbs = Math.max(...allCoords.map(Math.abs), 1);
    const scale = 5 / maxAbs;
    
    return positions.map(p => ({
      ...p,
      position: p.position.map(c => c * scale) as [number, number, number]
    }));
  }, [history, loadedVisualizations]);
  
  // Filter positions for current round
  const positions = useMemo(() => {
    if (!allRoundsPositions.length) return [];
    
    const roundIndex = Math.min(currentRound, history.length - 1);
    const filtered = allRoundsPositions.filter(p => p.roundIndex === roundIndex);
    
    return filtered;
  }, [allRoundsPositions, currentRound, history]);
  
  const maxRound = Math.max(0, history.length - 1);
  
  const handlePrevRound = () => {
    setCurrentRound(prev => Math.max(0, prev - 1));
  };
  
  const handleNextRound = () => {
    setCurrentRound(prev => Math.min(maxRound, prev + 1));
  };
  
  const handleExportGif = useCallback(async () => {
    if (!canvasRef.current || isExporting) return;
    
    setIsExporting(true);
    toast.info('Génération du GIF en cours...');
    
    try {
      const originalCanvas = canvasRef.current;
      const width = originalCanvas.width;
      const height = originalCanvas.height;
      
      console.log('Canvas dimensions:', width, height);
      
      const gif = new GIF({
        workers: 2,
        quality: 10,
        width,
        height,
        workerScript: '/gif.worker.js',
        background: '#000000'
      });
      
      // Capture each round with multiple frames for smooth transitions
      const framesPerRound = 5; // 5 frames per round for smoother animation
      
      for (let round = 0; round <= maxRound; round++) {
        setCurrentRound(round);
        
        // Capture multiple frames during and after transition
        for (let frame = 0; frame < framesPerRound; frame++) {
          await new Promise(resolve => setTimeout(resolve, 100));
          
          console.log(`Captured frame ${round * framesPerRound + frame + 1}/${(maxRound + 1) * framesPerRound}`);
          
          // Capture frame directly from WebGL canvas
          gif.addFrame(originalCanvas, { copy: true, delay: 200 }); // 200ms per frame = 1 second per round (5 frames)
        }
      }
      
      // Render GIF
      gif.on('finished', (blob: Blob) => {
        console.log('GIF rendering complete, size:', blob.size);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `federated-learning-3d-${Date.now()}.gif`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setIsExporting(false);
        toast.success('GIF exporté avec succès !');
      });
      
      gif.on('progress', (progress: number) => {
        console.log('GIF progress:', Math.round(progress * 100) + '%');
      });
      
      gif.render();
    } catch (error) {
      console.error('Error exporting GIF:', error);
      toast.error('Erreur lors de l\'export du GIF');
      setIsExporting(false);
    }
  }, [maxRound, isExporting]);
  
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
          <div className="w-3 h-3 bg-violet-500" />
          <span className="text-muted-foreground">Modèles Clusters (cubes)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-500" />
          <span className="text-muted-foreground">Modèles Clients (couleurs par cluster)</span>
        </div>
      </div>
      
      {/* 3D Canvas */}
      <div className="h-[400px] w-full rounded-lg overflow-hidden bg-black/50 border border-border">
        <Canvas
          camera={{ position: [8, 6, 8], fov: 50 }}
          style={{ background: 'transparent' }}
          gl={{ preserveDrawingBuffer: true }}
        >
          <CanvasCapture canvasRef={canvasRef} />
          <Scene positions={positions} />
        </Canvas>
      </div>
      
      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <Button 
          variant="outline" 
          size="icon"
          onClick={handlePrevRound}
          disabled={currentRound === 0 || isExporting}
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
          disabled={currentRound >= maxRound || isExporting}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        
        <Button 
          variant="outline"
          onClick={handleExportGif}
          disabled={isExporting || maxRound < 1}
          className="ml-2"
        >
          <Download className="h-4 w-4 mr-2" />
          {isExporting ? 'Export en cours...' : 'Export GIF'}
        </Button>
      </div>
      
      {/* Phase indicator */}
      <div className="flex justify-center gap-2 text-xs text-muted-foreground">
        <span>Utilisez la souris pour zoomer et faire pivoter</span>
      </div>
    </div>
  );
}
