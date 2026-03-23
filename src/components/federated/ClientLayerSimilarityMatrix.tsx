import React, { useEffect, useMemo, useState } from 'react';
import { RoundMetrics } from '@/lib/federated/types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface ClientLayerSimilarityMatrixProps {
  history: RoundMetrics[];
}

const valueToColor = (v: number) => {
  const hue = (1 - v) * 220;
  const lightness = 55 - v * 20;
  return `hsl(${hue},70%,${lightness}%)`;
};

const cosineSimilarity = (a: number[], b: number[]): number => {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
};

const ClientLayerSimilarityMatrix = ({ history }: ClientLayerSimilarityMatrixProps) => {
  const [selectedRound, setSelectedRound] = useState<number>(() => Math.max(0, (history?.length || 1) - 1));
  const [selectedLayer, setSelectedLayer] = useState<string>('0');

  useEffect(() => {
    setSelectedRound(Math.max(0, (history?.length || 1) - 1));
  }, [history.length]);

  const roundIdx = Math.min(Math.max(0, selectedRound), history.length - 1);
  const round = history[roundIdx];
  const clientMetrics = round?.clientMetrics;

  // Determine available layers from the first client that has weights
  const layerOptions = useMemo(() => {
    if (!clientMetrics) return [];
    const first = clientMetrics.find(cm => cm.weights?.layers?.length);
    if (!first?.weights) return [];
    return first.weights.layers.map((_, i) => ({ value: String(i), label: `Couche ${i + 1}` }));
  }, [clientMetrics]);

  // Compute pairwise cosine similarity matrix for selected layer
  const { matrix, clientNames } = useMemo(() => {
    if (!clientMetrics || layerOptions.length === 0) return { matrix: [] as number[][], clientNames: [] as string[] };
    const layerIdx = parseInt(selectedLayer);
    const validClients = clientMetrics.filter(cm => cm.weights?.layers?.[layerIdx]);
    const names = validClients.map(cm => cm.clientName);
    const vectors = validClients.map(cm => cm.weights!.layers[layerIdx]);
    const n = vectors.length;
    const mat: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        mat[i][j] = i === j ? 1 : cosineSimilarity(vectors[i], vectors[j]);
      }
    }
    return { matrix: mat, clientNames: names };
  }, [clientMetrics, selectedLayer, layerOptions]);

  if (!history.length || !clientMetrics?.length) {
    return (
      <div className="mt-6 p-4 rounded-lg bg-muted/10 border border-border text-sm text-muted-foreground">
        Aucune donnée client — lancez un round d'entraînement.
      </div>
    );
  }

  if (layerOptions.length === 0) {
    return (
      <div className="mt-6 p-4 rounded-lg bg-muted/10 border border-border text-sm text-muted-foreground">
        Les poids des modèles clients ne sont pas disponibles pour ce round.
      </div>
    );
  }

  const n = matrix.length;
  const cellSize = n <= 8 ? 28 : n <= 16 ? 20 : 14;

  return (
    <div className="mt-6">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium">Matrice de similarité cosinus (par couche)</h3>
        <div className="text-xs text-muted-foreground">Round {round.round + 1}</div>
      </div>

      <div className="p-3 rounded-lg bg-muted/20 border border-border">
        {/* Controls */}
        <div className="mb-3 flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Round :</label>
            <input
              type="range"
              min={0}
              max={Math.max(0, history.length - 1)}
              value={roundIdx}
              onChange={(e) => setSelectedRound(Number(e.target.value))}
              className="w-36"
            />
            <span className="text-xs">{roundIdx + 1} / {history.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Couche :</label>
            <Select value={selectedLayer} onValueChange={setSelectedLayer}>
              <SelectTrigger className="w-32 h-7 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {layerOptions.map(opt => (
                  <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Matrix */}
        {matrix.length > 0 && (
          <div className="flex gap-4">
            <div className="flex-shrink-0">
              <div className="overflow-auto" style={{ maxWidth: Math.min(600, n * (cellSize + 2)), maxHeight: 420 }}>
                {/* Column labels */}
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 2 }}>
                  {clientNames.map((name, i) => (
                    <div key={`col-${i}`} className="text-[10px] text-center text-muted-foreground truncate" style={{ width: cellSize }}>
                      {name.split(' ').slice(0, 2).join(' ')}
                    </div>
                  ))}
                </div>
                <div style={{ height: 6 }} />
                {/* Cells */}
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 2 }}>
                  {matrix.flatMap((row, i) => row.map((v, j) => {
                    const normalized = (v + 1) / 2; // map [-1,1] to [0,1] for color
                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        title={`${clientNames[i]} ↔ ${clientNames[j]}: ${v.toFixed(4)}`}
                        style={{
                          width: cellSize,
                          height: cellSize,
                          backgroundColor: valueToColor(normalized),
                          border: '1px solid rgba(0,0,0,0.06)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <span style={{ fontSize: n <= 8 ? '9px' : '7px', fontWeight: 500 }}>
                          {v.toFixed(2)}
                        </span>
                      </div>
                    );
                  }))}
                </div>
              </div>
            </div>

            {/* Legend */}
            <div className="flex-1">
              <div className="mb-2 text-xs text-muted-foreground">Légende</div>
              <div className="flex items-center gap-2">
                {(() => {
                  const stops = [0, 0.25, 0.5, 0.75, 1].map(p => `${valueToColor(p)} ${Math.round(p * 100)}%`).join(', ');
                  return <div style={{ width: 120, height: 12, background: `linear-gradient(90deg, ${stops})`, borderRadius: 4 }} />;
                })()}
                <span className="text-xs">-1</span>
                <span className="text-xs">0</span>
                <span className="text-xs">1</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClientLayerSimilarityMatrix;
