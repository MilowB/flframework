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
  Area,
  ComposedChart,
} from 'recharts';
import { Server, Layers, Users } from 'lucide-react';
import type { ExperimentData } from '@/lib/federated/results/experimentStorage';

interface ComparisonChartProps {
  experiments: { name: string; data: ExperimentData }[];
}

// Color palette for experiments (extended for more than 3)
const EXP_COLORS = [
  'hsl(174, 72%, 56%)', // Cyan
  'hsl(262, 83%, 58%)', // Purple
  'hsl(38, 92%, 50%)', // Orange
  'hsl(340, 82%, 52%)', // Pink
  'hsl(120, 60%, 45%)', // Green
  'hsl(200, 80%, 55%)', // Blue
  'hsl(45, 90%, 55%)', // Yellow
  'hsl(280, 70%, 60%)', // Violet
];

const getExpColor = (idx: number) => EXP_COLORS[idx % EXP_COLORS.length];

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

  // Cluster average + STD per experiment
  const clusterChartData = useMemo(() => {
    const maxRounds = Math.max(...experiments.map((e) => e.data.roundHistory.length));
    const data: Record<string, number | string>[] = [];

    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: Record<string, number | string> = { round: round + 1 };
      experiments.forEach((exp, expIdx) => {
        const h = exp.data.roundHistory[round];
        if (h && h.clusterMetrics && h.clusterMetrics.length > 0) {
          const accuracies = h.clusterMetrics.map((cm) => cm.accuracy * 100);
          const mean = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
          const variance = accuracies.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / accuracies.length;
          const std = Math.sqrt(variance);
          
          dataPoint[`exp${expIdx}_cluster_mean`] = mean;
          dataPoint[`exp${expIdx}_cluster_std`] = std;
          dataPoint[`exp${expIdx}_cluster_min`] = mean - std;
          dataPoint[`exp${expIdx}_cluster_max`] = mean + std;
        }
      });
      data.push(dataPoint);
    }
    return data;
  }, [experiments]);

  // Client average + STD per experiment
  const clientChartData = useMemo(() => {
    const maxRounds = Math.max(...experiments.map((e) => e.data.roundHistory.length));
    const data: Record<string, number | string>[] = [];

    for (let round = 0; round < maxRounds; round++) {
      const dataPoint: Record<string, number | string> = { round: round + 1 };
      experiments.forEach((exp, expIdx) => {
        const h = exp.data.roundHistory[round];
        if (h && h.clientMetrics && h.clientMetrics.length > 0) {
          const accuracies = h.clientMetrics.map((cm) => cm.testAccuracy * 100);
          const mean = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
          const variance = accuracies.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / accuracies.length;
          const std = Math.sqrt(variance);
          
          dataPoint[`exp${expIdx}_client_mean`] = mean;
          dataPoint[`exp${expIdx}_client_std`] = std;
          dataPoint[`exp${expIdx}_client_min`] = mean - std;
          dataPoint[`exp${expIdx}_client_max`] = mean + std;
        }
      });
      data.push(dataPoint);
    }
    return data;
  }, [experiments]);

  const hasClusterData = experiments.some(e => 
    e.data.roundHistory.some(h => h.clusterMetrics && h.clusterMetrics.length > 0)
  );

  const hasClientData = experiments.some(e => 
    e.data.roundHistory.some(h => h.clientMetrics && h.clientMetrics.length > 0)
  );

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Comparaison des métriques</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="server" className="w-full">
          <TabsList className="w-full grid grid-cols-3 bg-muted/30 mb-4">
            <TabsTrigger value="server" className="gap-2">
              <Server className="w-4 h-4" />
              Serveur (Global)
            </TabsTrigger>
            <TabsTrigger value="clusters" className="gap-2">
              <Layers className="w-4 h-4" />
              Clusters
            </TabsTrigger>
            <TabsTrigger value="clients" className="gap-2">
              <Users className="w-4 h-4" />
              Clients
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
                          stroke={getExpColor(idx)}
                          strokeWidth={2}
                          dot={{ fill: getExpColor(idx), strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: getExpColor(idx) }}
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
                          stroke={getExpColor(idx)}
                          strokeWidth={2}
                          dot={{ fill: getExpColor(idx), strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: getExpColor(idx) }}
                          connectNulls
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Clusters comparison - Average with STD band */}
          <TabsContent value="clusters">
            {hasClusterData ? (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-3">
                  Précision moyenne des clusters (± écart-type)
                </h4>
                <div className="h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={clusterChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
                          if (name.includes('_cluster_mean')) {
                            const expIdx = parseInt(name.replace('exp', '').replace('_cluster_mean', ''));
                            return [`${value.toFixed(1)}%`, `${experiments[expIdx]?.name} (μ)`];
                          }
                          return [null, null];
                        }}
                      />
                      <Legend
                        wrapperStyle={{ paddingTop: '10px' }}
                        formatter={(value) => {
                          const expIdx = parseInt(value.replace('exp', '').replace('_cluster_mean', ''));
                          return `${experiments[expIdx]?.name} (μ ± σ)`;
                        }}
                      />
                      {experiments.map((_, idx) => (
                        <Area
                          key={`cluster-area-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_cluster_max`}
                          stroke="none"
                          fill={getExpColor(idx)}
                          fillOpacity={0.15}
                          connectNulls
                          legendType="none"
                        />
                      ))}
                      {experiments.map((_, idx) => (
                        <Area
                          key={`cluster-area-min-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_cluster_min`}
                          stroke="none"
                          fill="hsl(222, 47%, 10%)"
                          fillOpacity={1}
                          connectNulls
                          legendType="none"
                        />
                      ))}
                      {experiments.map((_, idx) => (
                        <Line
                          key={`cluster-mean-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_cluster_mean`}
                          stroke={getExpColor(idx)}
                          strokeWidth={2}
                          dot={{ fill: getExpColor(idx), strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: getExpColor(idx) }}
                          connectNulls
                        />
                      ))}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                Aucune donnée de cluster disponible
              </div>
            )}
          </TabsContent>

          {/* Clients comparison - Average with STD band */}
          <TabsContent value="clients">
            {hasClientData ? (
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-3">
                  Précision moyenne des clients (± écart-type)
                </h4>
                <div className="h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={clientChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
                          if (name.includes('_client_mean')) {
                            const expIdx = parseInt(name.replace('exp', '').replace('_client_mean', ''));
                            return [`${value.toFixed(1)}%`, `${experiments[expIdx]?.name} (μ)`];
                          }
                          return [null, null];
                        }}
                      />
                      <Legend
                        wrapperStyle={{ paddingTop: '10px' }}
                        formatter={(value) => {
                          const expIdx = parseInt(value.replace('exp', '').replace('_client_mean', ''));
                          return `${experiments[expIdx]?.name} (μ ± σ)`;
                        }}
                      />
                      {experiments.map((_, idx) => (
                        <Area
                          key={`client-area-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_client_max`}
                          stroke="none"
                          fill={getExpColor(idx)}
                          fillOpacity={0.15}
                          connectNulls
                          legendType="none"
                        />
                      ))}
                      {experiments.map((_, idx) => (
                        <Area
                          key={`client-area-min-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_client_min`}
                          stroke="none"
                          fill="hsl(222, 47%, 10%)"
                          fillOpacity={1}
                          connectNulls
                          legendType="none"
                        />
                      ))}
                      {experiments.map((_, idx) => (
                        <Line
                          key={`client-mean-${idx}`}
                          type="monotone"
                          dataKey={`exp${idx}_client_mean`}
                          stroke={getExpColor(idx)}
                          strokeWidth={2}
                          dot={{ fill: getExpColor(idx), strokeWidth: 0, r: 3 }}
                          activeDot={{ r: 5, fill: getExpColor(idx) }}
                          connectNulls
                        />
                      ))}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                Aucune donnée client disponible
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};