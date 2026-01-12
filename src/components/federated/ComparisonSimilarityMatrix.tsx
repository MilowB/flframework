import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import type { ExperimentData } from '@/lib/federated/results/experimentStorage';

interface ComparisonSimilarityMatrixProps {
  experiments: { name: string; data: ExperimentData }[];
}

const palette = [
  '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a'
];

const valueToColor = (v: number) => {
  const hue = (1 - v) * 220;
  const lightness = 55 - v * 20;
  return `hsl(${hue},70%,${lightness}%)`;
};

export const ComparisonSimilarityMatrix = ({ experiments }: ComparisonSimilarityMatrixProps) => {
  const [selectedRound, setSelectedRound] = useState<number>(0);
  
  const maxRounds = useMemo(() => 
    Math.max(...experiments.map(e => e.data.roundHistory.length), 1) - 1,
    [experiments]
  );

  const roundIdx = Math.min(selectedRound, maxRounds);

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Matrices de similarité</CardTitle>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">Round {roundIdx + 1} / {maxRounds + 1}</span>
            <div className="w-48">
              <Slider
                value={[roundIdx]}
                min={0}
                max={maxRounds}
                step={1}
                onValueChange={([v]) => setSelectedRound(v)}
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {experiments.map((exp, expIdx) => (
            <SingleMatrix
              key={expIdx}
              name={exp.name}
              expIdx={expIdx}
              roundData={exp.data.roundHistory[roundIdx]}
            />
          ))}
        </div>
        
        {/* Legend */}
        <div className="mt-6 flex items-center justify-center gap-4">
          <span className="text-xs text-muted-foreground">Similarité:</span>
          <div className="flex items-center gap-2">
            {(() => {
              const stops = [0, 0.25, 0.5, 0.75, 1].map(p => `${valueToColor(p)} ${Math.round(p * 100)}%`).join(', ');
              const gradient = `linear-gradient(90deg, ${stops})`;
              return (
                <div style={{ width: 160, height: 12, background: gradient, borderRadius: 4 }} />
              );
            })()}
            <span className="text-xs">0%</span>
            <span className="text-xs">100%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

interface SingleMatrixProps {
  name: string;
  expIdx: number;
  roundData: ExperimentData['roundHistory'][0] | undefined;
}

const SingleMatrix = ({ name, expIdx, roundData }: SingleMatrixProps) => {
  if (!roundData || !roundData.distanceMatrix || roundData.distanceMatrix.length === 0) {
    return (
      <div className="p-4 rounded-lg bg-muted/10 border border-border">
        <h4 className="text-sm font-medium mb-2">{name}</h4>
        <div className="text-sm text-muted-foreground text-center py-8">
          Pas de matrice disponible
        </div>
      </div>
    );
  }

  const D = roundData.distanceMatrix;
  const participating = roundData.participatingClients || [];
  const clusters = roundData.clusters || [];

  // Convert distances to similarity via RBF
  const mean = useMemo(() => {
    let sum = 0; let count = 0;
    for (let i = 0; i < D.length; i++) {
      for (let j = i + 1; j < D.length; j++) {
        sum += D[i][j];
        count++;
      }
    }
    return count > 0 ? sum / count : 1;
  }, [D]);
  
  const sigma = mean || 1;

  const S: number[][] = useMemo(() => {
    const result: number[][] = Array.from({ length: D.length }, () => new Array(D.length).fill(0));
    for (let i = 0; i < D.length; i++) {
      for (let j = 0; j < D.length; j++) {
        if (i === j) { result[i][j] = 1; continue; }
        result[i][j] = Math.exp(-D[i][j] / sigma);
      }
    }
    return result;
  }, [D, sigma]);

  // Sort participating client ids
  const sortedIds = [...participating].sort((a, b) => a.localeCompare(b));
  const idToIndex = new Map<string, number>();
  participating.forEach((id, i) => idToIndex.set(id, i));

  // Reorder similarity matrix
  const Sordered: number[][] = useMemo(() => {
    const result: number[][] = Array.from({ length: S.length }, () => new Array(S.length).fill(0));
    for (let i = 0; i < sortedIds.length; i++) {
      for (let j = 0; j < sortedIds.length; j++) {
        const oi = idToIndex.get(sortedIds[i]) ?? i;
        const oj = idToIndex.get(sortedIds[j]) ?? j;
        result[i][j] = S[oi][oj];
      }
    }
    return result;
  }, [S, sortedIds, idToIndex]);

  const n = Sordered.length;
  const cellSize = n <= 8 ? 20 : (n <= 16 ? 14 : 10);

  return (
    <div className="p-4 rounded-lg bg-muted/10 border border-border">
      <h4 className="text-sm font-medium mb-3">{name}</h4>
      
      <div className="overflow-auto" style={{ maxHeight: 280 }}>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 1 }}>
          {Sordered.flatMap((row, i) => row.map((v, j) => (
            <div
              key={`cell-${i}-${j}`}
              title={`${(v*100).toFixed(1)}%`}
              style={{
                width: cellSize,
                height: cellSize,
                backgroundColor: valueToColor(v),
                border: '1px solid rgba(0,0,0,0.06)'
              }}
            >
              <span style={{fontSize: '11px', fontWeight: 500}}>{(v*100).toFixed(1)}</span>
            </div>
          )))}
        </div>
      </div>

      {/* Clusters info */}
      {clusters.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {clusters.map((grp, gidx) => (
            <div key={gidx} className="flex items-center gap-1">
              <div 
                style={{ 
                  width: 10, 
                  height: 10, 
                  backgroundColor: palette[gidx % palette.length], 
                  borderRadius: 2 
                }} 
              />
              <span className="text-xs text-muted-foreground">
                G{gidx + 1} ({grp.length})
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Silhouette */}
      {roundData.silhouetteAvg !== undefined && (
        <div className="mt-2 text-xs text-muted-foreground">
          Silhouette: {(roundData.silhouetteAvg * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
};