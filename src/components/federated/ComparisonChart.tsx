import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Server, Layers, Users } from 'lucide-react';
import type { ExperimentData } from '@/lib/federated/results/experimentStorage';

interface ComparisonChartProps {
  experiments: { name: string; data: ExperimentData }[];
}

// Color palette for experiments
const EXP_COLORS = [
  'hsl(174, 72%, 56%)', // Cyan
  'hsl(262, 83%, 58%)', // Purple
  'hsl(38, 92%, 50%)', // Orange
];

// Color palette for clusters/clients within each experiment
const CLUSTER_COLORS = [
  ['hsl(174, 72%, 56%)', 'hsl(174, 72%, 40%)', 'hsl(174, 72%, 70%)'],
  ['hsl(262, 83%, 58%)', 'hsl(262, 83%, 42%)', 'hsl(262, 83%, 72%)'],
  ['hsl(38, 92%, 50%)', 'hsl(38, 92%, 35%)', 'hsl(38, 92%, 65%)'],
];

export const ComparisonChart = ({ experiments }: ComparisonChartProps) => {
  // Merge server data from all experiments
  const serverChartData = useMemo(() => {
    const maxRounds = Math.max(...experiments.map((e) => e.data.roundHistory.length));
    const data: Record<string, number | string>[] = [];

    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: Record<string, number | string> = { round: round + 1 };
      experiments.forEach((exp, idx) => {
        const h = exp.data.roundHistory[round];
        if (h) {
          dataPoint[`exp${idx}_loss`] = h.globalLoss;
          dataPoint[`exp${idx}_acc`] = h.globalAccuracy * 100;
        }
      });
      data.push(dataPoint);
    }
    return data;
  }, [experiments]);

  // Get all unique cluster IDs across all experiments
  const allClusters = useMemo(() => {
    const clusterMap = new Map<number, Set<number>>();
    experiments.forEach((exp, expIdx) => {
      exp.data.roundHistory.forEach((h) => {
        if (h.clusterMetrics) {
          h.clusterMetrics.forEach((cm) => {
            if (!clusterMap.has(expIdx)) clusterMap.set(expIdx, new Set());
            clusterMap.get(expIdx)!.add(cm.clusterId);
          });
        }
      });
    });
    return clusterMap;
  }, [experiments]);

  // Merge cluster data from all experiments
  const clusterChartData = useMemo(() => {
    const maxRounds = Math.max(...experiments.map((e) => e.data.roundHistory.length));
    const data: Record<string, number | string>[] = [];

    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: Record<string, number | string> = { round: round + 1 };
      experiments.forEach((exp, expIdx) => {
        const h = exp.data.roundHistory[round];
        if (h && h.clusterMetrics) {
          h.clusterMetrics.forEach((cm) => {
            dataPoint[`exp${expIdx}_c${cm.clusterId}`] = cm.accuracy * 100;
          });
        }
      });
      data.push(dataPoint);
    }
    return data;
  }, [experiments]);

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Comparaison des métriques</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="server" className="w-full">
          <TabsList className="w-full grid grid-cols-2 bg-muted/30 mb-4">
            <TabsTrigger value="server" className="gap-2">
              <Server className="w-4 h-4" />
              Serveur (Global)
            </TabsTrigger>
            <TabsTrigger value="clusters" className="gap-2">
              <Layers className="w-4 h-4" />
              Clusters
            </TabsTrigger>
          </TabsList>

          {/* Server comparison */}
          <TabsContent value="server">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Loss chart */}
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-3">Loss Globale</h4>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={serverChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
                      <XAxis
                        dataKey="round"
                        stroke="hsl(215, 20%, 55%)"
                        fontSize={12}
                        tickLine={false}
                      />
                      <YAxis
                        stroke="hsl(215, 20%, 55%)"
                        fontSize={12}
                        tickLine={false}
                        domain={[0, 'auto']}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'hsl(222, 47%, 10%)',
                          border: '1px solid hsl(222, 30%, 18%)',
                          borderRadius: '8px',
                          color: 'hsl(210, 40%, 98%)',
                        }}
                        labelFormatter={(label) => `Round ${label}`}
                        formatter={(value: number, name: string) => {
                          const expIdx = parseInt(name.replace('exp', '').replace('_loss', ''));
                          return [value.toFixed(4), experiments[expIdx]?.name || name];
                        }}
                      />
                      <Legend
                        wrapperStyle={{ paddingTop: '10px' }}
                        formatter={(value) => {
                          const expIdx = parseInt(value.replace('exp', '').replace('_loss', ''));
                          return experiments[expIdx]?.name || value;
                        }}
                      />
                      {experiments.map((_, idx) => (
                        <Line
                          key={`loss-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_loss`}
                          stroke={EXP_COLORS[idx]}
                          strokeWidth={2}
                          dot={{ fill: EXP_COLORS[idx], strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: EXP_COLORS[idx] }}
                          connectNulls
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Accuracy chart */}
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-3">Précision Globale</h4>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={serverChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
                      <XAxis
                        dataKey="round"
                        stroke="hsl(215, 20%, 55%)"
                        fontSize={12}
                        tickLine={false}
                      />
                      <YAxis
                        stroke="hsl(215, 20%, 55%)"
                        fontSize={12}
                        tickLine={false}
                        domain={[0, 100]}
                        tickFormatter={(value) => `${value}%`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'hsl(222, 47%, 10%)',
                          border: '1px solid hsl(222, 30%, 18%)',
                          borderRadius: '8px',
                          color: 'hsl(210, 40%, 98%)',
                        }}
                        labelFormatter={(label) => `Round ${label}`}
                        formatter={(value: number, name: string) => {
                          const expIdx = parseInt(name.replace('exp', '').replace('_acc', ''));
                          return [`${value.toFixed(1)}%`, experiments[expIdx]?.name || name];
                        }}
                      />
                      <Legend
                        wrapperStyle={{ paddingTop: '10px' }}
                        formatter={(value) => {
                          const expIdx = parseInt(value.replace('exp', '').replace('_acc', ''));
                          return experiments[expIdx]?.name || value;
                        }}
                      />
                      {experiments.map((_, idx) => (
                        <Line
                          key={`acc-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_acc`}
                          stroke={EXP_COLORS[idx]}
                          strokeWidth={2}
                          dot={{ fill: EXP_COLORS[idx], strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: EXP_COLORS[idx] }}
                          connectNulls
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Clusters comparison */}
          <TabsContent value="clusters">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {experiments.map((exp, expIdx) => {
                const clusters = allClusters.get(expIdx);
                return (
                  <div key={expIdx}>
                    <h4 className="text-sm font-medium text-foreground mb-3">
                      {exp.name}
                    </h4>
                    {clusters && clusters.size > 0 ? (
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={clusterChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
                            <XAxis
                              dataKey="round"
                              stroke="hsl(215, 20%, 55%)"
                              fontSize={11}
                              tickLine={false}
                            />
                            <YAxis
                              stroke="hsl(215, 20%, 55%)"
                              fontSize={11}
                              tickLine={false}
                              domain={[0, 100]}
                              tickFormatter={(value) => `${value}%`}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: 'hsl(222, 47%, 10%)',
                                border: '1px solid hsl(222, 30%, 18%)',
                                borderRadius: '8px',
                                color: 'hsl(210, 40%, 98%)',
                              }}
                              labelFormatter={(label) => `Round ${label}`}
                              formatter={(value: number, name: string) => {
                                const clusterId = name.split('_c')[1];
                                return [`${value.toFixed(1)}%`, `Cluster #${parseInt(clusterId) + 1}`];
                              }}
                            />
                            <Legend
                              wrapperStyle={{ paddingTop: '5px', fontSize: '10px' }}
                              formatter={(value) => {
                                const clusterId = value.split('_c')[1];
                                return `C${parseInt(clusterId) + 1}`;
                              }}
                            />
                            {Array.from(clusters).map((clusterId, cIdx) => (
                              <Line
                                key={`exp${expIdx}_c${clusterId}`}
                                type="monotone"
                                dataKey={`exp${expIdx}_c${clusterId}`}
                                stroke={CLUSTER_COLORS[expIdx]?.[cIdx % 3] || EXP_COLORS[expIdx]}
                                strokeWidth={2}
                                dot={{ fill: CLUSTER_COLORS[expIdx]?.[cIdx % 3] || EXP_COLORS[expIdx], strokeWidth: 0, r: 2 }}
                                activeDot={{ r: 4 }}
                                connectNulls
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="h-[250px] flex items-center justify-center text-muted-foreground text-sm">
                        Pas de clusters
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
