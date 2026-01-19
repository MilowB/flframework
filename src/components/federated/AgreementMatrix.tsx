import React, { useEffect, useState } from 'react';
import { RoundMetrics } from '@/lib/federated/types';

interface AgreementMatrixProps {
  history: RoundMetrics[];
  numRuns?: number; // Number of clustering runs used (default: 20)
}

const palette = [
  '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a'
];

// Color scale for agreement matrix (0 to numRuns)
const agreementToColor = (value: number, maxValue: number) => {
  const normalized = maxValue > 0 ? value / maxValue : 0;
  // Blue (low agreement) to Red (high agreement)
  const hue = (1 - normalized) * 220;
  const lightness = 55 - normalized * 20;
  return `hsl(${hue}, 70%, ${lightness}%)`;
};

export const AgreementMatrix = ({ history, numRuns = 20 }: AgreementMatrixProps) => {
  const [selected, setSelected] = useState<number>(() => Math.max(0, (history?.length || 1) - 1));

  // Keep selected in sync with history updates
  useEffect(() => {
    setSelected(Math.max(0, (history?.length || 1) - 1));
  }, [history.length]);

  if (!history || history.length === 0) {
    return (
      <div className="mt-4 p-4 rounded-lg bg-muted/10 border border-border text-sm text-muted-foreground">
        Aucune matrice d'agreement disponible — lancez un round avec la matrice d'agreement activée.
      </div>
    );
  }

  const idx = Math.min(Math.max(0, selected), history.length - 1);
  const round = history[idx];
  const A = round.agreementMatrix;
  const participating = round.participatingClients || [];

  if (!A || A.length === 0) {
    return (
      <div className="mt-4 p-4 rounded-lg bg-muted/10 border border-border text-sm text-muted-foreground">
        Matrice d'agreement non disponible — activez l'option "Matrice d'agreement" dans le Serveur Central.
      </div>
    );
  }

  // Build cluster color map
  const clusters = round.clusters || [];
  const clusterColorById = new Map<string, string>();
  clusters.forEach((group, gidx) => {
    const color = palette[gidx % palette.length];
    group.forEach(id => clusterColorById.set(id, color));
  });

  // Sort participating client ids ascending
  const sortedIds = [...participating].slice().sort((a, b) => a.localeCompare(b));
  const idToIndex = new Map<string, number>();
  participating.forEach((id, i) => idToIndex.set(id, i));

  // Reorder agreement matrix to match sortedIds order
  const Aordered: number[][] = Array.from({ length: A.length }, () => new Array(A.length).fill(0));
  for (let i = 0; i < sortedIds.length; i++) {
    for (let j = 0; j < sortedIds.length; j++) {
      const oi = idToIndex.get(sortedIds[i]) ?? i;
      const oj = idToIndex.get(sortedIds[j]) ?? j;
      Aordered[i][j] = A[oi]?.[oj] ?? 0;
    }
  }

  const n = Aordered.length;
  const cellSize = n <= 8 ? 32 : (n <= 16 ? 24 : 18);

  return (
    <div className="mt-6">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium">Matrice d'agreement</h3>
        <div className="text-xs text-muted-foreground">Round {round.round + 1}</div>
      </div>

      <div className="p-3 rounded-lg bg-muted/20 border border-border">
        <div className="mb-3 flex items-center gap-3">
          <label className="text-xs text-muted-foreground">Explorer les rounds :</label>
          <input
            type="range"
            min={0}
            max={Math.max(0, history.length - 1)}
            value={idx}
            onChange={(e) => setSelected(Number(e.target.value))}
            className="w-48"
          />
          <div className="text-xs">Round {idx + 1} / {history.length}</div>
        </div>

        <div className="flex gap-4">
          <div className="flex-shrink-0">
            <div className="overflow-auto" style={{ maxWidth: Math.min(600, n * (cellSize + 2)), maxHeight: 420 }}>
              {/* Column labels */}
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 2 }}>
                {sortedIds.map((id, i) => (
                  <div 
                    key={`col-label-${i}`} 
                    className="text-[9px] text-center text-muted-foreground truncate" 
                    style={{ width: cellSize }}
                    title={id}
                  >
                    {id ? id.replace('client-', 'C') : `C${i}`}
                  </div>
                ))}
              </div>
              <div style={{ height: 4 }} />
              {/* Matrix cells */}
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 2 }}>
                {Aordered.flatMap((row, i) => row.map((v, j) => (
                  <div
                    key={`cell-${i}-${j}`}
                    title={`${sortedIds[i]} ↔ ${sortedIds[j]}: ${v}/${numRuns} (${((v / numRuns) * 100).toFixed(0)}%)`}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: agreementToColor(v, numRuns),
                      border: '1px solid rgba(0,0,0,0.06)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <span style={{ fontSize: '10px', fontWeight: 500, color: v > numRuns * 0.6 ? '#fff' : '#333' }}>
                      {v}
                    </span>
                  </div>
                )))}
              </div>
            </div>
          </div>

          <div className="flex-1">
            <div className="mb-3 text-xs text-muted-foreground">Clusters finaux</div>
            <div className="flex flex-wrap gap-2 mb-4 relative">
              {clusters.length === 0 ? (
                <div className="text-sm text-muted-foreground">Aucun cluster</div>
              ) : clusters.map((grp, gidx) => (
                <LegendGroup
                  key={`grp-${gidx}`}
                  idx={gidx}
                  grp={grp}
                  color={palette[gidx % palette.length]}
                />
              ))}
            </div>

            <div className="mb-2 text-xs text-muted-foreground">Légende Agreement</div>
            <div className="flex items-center gap-2">
              {(() => {
                const stops = [0, 0.25, 0.5, 0.75, 1].map(p => 
                  `${agreementToColor(p * numRuns, numRuns)} ${Math.round(p * 100)}%`
                ).join(', ');
                const gradient = `linear-gradient(90deg, ${stops})`;
                return (
                  <div style={{ width: 140, height: 12, background: gradient, borderRadius: 4 }} />
                );
              })()}
              <div className="text-xs">0</div>
              <div className="text-xs">{Math.round(numRuns / 2)}</div>
              <div className="text-xs">{numRuns}</div>
            </div>
            <div className="mt-1 text-[10px] text-muted-foreground">
              Nombre de fois que deux clients ont été clusterisés ensemble sur {numRuns} runs
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgreementMatrix;

// Helper component for legend groups with hover tooltip
const LegendGroup = ({ idx, grp, color }: { idx: number; grp: string[]; color: string }) => {
  const [hover, setHover] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="relative"
    >
      <div className="flex items-center gap-2 pr-3 cursor-default">
        <div style={{ width: 12, height: 12, backgroundColor: color, borderRadius: 3 }} />
        <div className="text-sm">Groupe {idx + 1} ({grp.length})</div>
      </div>

      {hover && (
        <div className="absolute z-50 mt-2 left-0 w-48 p-2 bg-popover border border-border rounded shadow-lg text-xs">
          <div className="font-medium mb-1">Membres (Groupe {idx + 1})</div>
          <div className="max-h-40 overflow-auto">
            {grp.map((id) => (
              <div key={id} className="truncate">{id}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

