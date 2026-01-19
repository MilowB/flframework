import React, { useEffect, useMemo, useState } from 'react';
import { RoundMetrics } from '@/lib/federated/types';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface SimilarityMatrixProps {
	history: RoundMetrics[];
}

const palette = [
	'#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a'
];

const valueToColor = (v: number) => {
	// v in [0,1]
	const hue = (1 - v) * 220; // blue->red-ish range
	const lightness = 55 - v * 20; // darker when high similarity
	return `hsl(${hue},70%,${lightness}%)`;
};

export const SimilarityMatrix = ({ history }: SimilarityMatrixProps) => {
	const [selected, setSelected] = useState<number>(() => Math.max(0, (history?.length || 1) - 1));

	// Keep selected in sync with history updates: default to last round
	useEffect(() => {
		setSelected(Math.max(0, (history?.length || 1) - 1));
	}, [history.length]);

	if (!history || history.length === 0) {
		return (
			<div className="mt-4 p-4 rounded-lg bg-muted/10 border border-border text-sm text-muted-foreground">
				Aucune matrice de similarité disponible — lancez un round.
			</div>
		);
	}

	const idx = Math.min(Math.max(0, selected), history.length - 1);
	const round = history[idx];
	const D = round.distanceMatrix;
	const participating = round.participatingClients || [];

	if (!D || D.length === 0) {
		return (
			<div className="mt-4 p-4 rounded-lg bg-muted/10 border border-border text-sm text-muted-foreground">
				La matrice de distances n'est pas disponible pour le round sélectionné.
			</div>
		);
	}

	// Convert distances to similarity via RBF (same formula as server)
	const { mean } = useMemo(() => {
		let sum = 0; let count = 0;
		for (let i = 0; i < D.length; i++) for (let j = i + 1; j < D.length; j++) { sum += D[i][j]; count++; }
		return { mean: count > 0 ? sum / count : 1 };
	}, [D]);
	const sigma = mean || 1;

	const S: number[][] = Array.from({ length: D.length }, () => new Array(D.length).fill(0));
	for (let i = 0; i < D.length; i++) {
		for (let j = 0; j < D.length; j++) {
			if (i === j) { S[i][j] = 1; continue; }
			S[i][j] = Math.exp(-D[i][j] / sigma);
		}
	}

	// build cluster color map
	const clusters = round.clusters || [];
	const clusterColorById = new Map<string, string>();
	clusters.forEach((group, gidx) => {
		const color = palette[gidx % palette.length];
		group.forEach(id => clusterColorById.set(id, color));
	});

	// Use participating clients in their original order (already sorted in simulation.ts)
	const sortedIds = participating;
	
	// No need to reorder - the distance matrix is already in the correct order
	const Sordered = S;

	const n = Sordered.length;
	const cellSize = n <= 8 ? 28 : (n <= 16 ? 20 : 14); // adaptive sizing

	return (
		<div className="mt-6">
			<div className="mb-3 flex items-center justify-between">
				<h3 className="text-sm font-medium">Matrice de similarité</h3>
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
							<div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 2 }}>
								{sortedIds.map((id, i) => (
									<div key={`col-label-${i}`} className="text-[10px] text-center text-muted-foreground" style={{ width: cellSize }}>
										{id ? id.split(' ').slice(0,2).join(' ') : `C${i+1}`}
									</div>
								))}
							</div>
							<div style={{ height: 8 }} />
							<div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, ${cellSize}px)`, gap: 2 }}>
								{Sordered.flatMap((row, i) => row.map((v, j) => (
									<div
										key={`cell-${i}-${j}`}
										title={`(${i+1},${j+1}) ${(v*100).toFixed(1)}%`}
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
					</div>

					<div className="flex-1">
						<div className="mb-3 text-xs text-muted-foreground">Clusters</div>
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

						<div className="mb-2 text-xs text-muted-foreground">Légende Similarité</div>
						<div className="flex items-center gap-3">
							{(() => {
								// Build gradient stops that match valueToColor so legend matches cells
								const stops = [0, 0.25, 0.5, 0.75, 1].map(p => `${valueToColor(p)} ${Math.round(p * 100)}%`).join(', ');
								const gradient = `linear-gradient(90deg, ${stops})`;
								return (
									<div style={{ width: 160, height: 12, background: gradient, borderRadius: 4 }} />
								);
							})()}
							<div className="text-xs">0%</div>
							<div className="text-xs">50%</div>
							<div className="text-xs">100%</div>
						</div>
					</div>
				</div>

				{/* Silhouette chart (moved below into its own panel) */}

				{/* Silhouette panel placed below the similarity matrix */}
				<div className="mt-4 p-3 rounded-lg bg-muted/20 border border-border">
					<div className="mb-2 text-sm font-medium">Silhouette moyenne par round</div>
					<div style={{ width: '100%', height: 120 }}>
						<ResponsiveContainer width="100%" height="100%">
							<LineChart data={history.map((h, i) => ({ round: i + 1, silhouette: (h.silhouetteAvg ?? 0) * 100 }))}>
								<CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
								<XAxis dataKey="round" stroke="hsl(215, 20%, 55%)" tick={{ fontSize: 12 }} />
								<YAxis stroke="hsl(215, 20%, 55%)" tick={{ fontSize: 12 }} unit="%" />
								<Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
								<Line type="monotone" dataKey="silhouette" stroke="#8884d8" strokeWidth={2} dot={{ r: 3 }} />
							</LineChart>
						</ResponsiveContainer>
					</div>
				</div>
			</div>
		</div>
	);
};

export default SimilarityMatrix;

// Small helper component to render a legend group with hover tooltip showing members
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

