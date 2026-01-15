import { useMemo } from 'react';
import { RoundMetrics, ModelWeights } from '@/lib/federated/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingDown, TrendingUp, Server, Layers, Users, Box } from 'lucide-react';
import SimilarityMatrix from './SimilarityMatrix';
import { ModelVisualization3D } from './ModelVisualization3D';
import { Model3DPosition } from '@/lib/federated/visualization/pca';

interface MetricsChartProps {
  history: RoundMetrics[];
  clientModels?: Map<string, { weights: { layers: number[][]; bias: number[] }; name: string }>;
  clusterModels?: Map<string, { layers: number[][]; bias: number[] }>;
  globalModel?: ModelWeights | null;
  loadedVisualizations?: { round: number; models: Model3DPosition[] }[];
}

// Color palette for multiple lines
const COLORS = [
  'hsl(262, 83%, 58%)', // Purple
  'hsl(174, 72%, 56%)', // Cyan
  'hsl(142, 72%, 45%)', // Green
  'hsl(38, 92%, 50%)',  // Orange
  'hsl(0, 72%, 50%)',   // Red
  'hsl(210, 72%, 50%)', // Blue
  'hsl(320, 72%, 50%)', // Pink
  'hsl(60, 72%, 50%)',  // Yellow
];

export const MetricsChart = ({ history, clientModels, clusterModels, globalModel, loadedVisualizations }: MetricsChartProps) => {
  // Server chart data (global model)
  const serverChartData = useMemo(() => history.map((h) => ({
    round: h.round + 1,
    loss: h.globalLoss,
    accuracy: h.globalAccuracy * 100,
  })), [history]);

  // Get all unique cluster IDs across all rounds
  const availableClusters = useMemo(() => {
    const clusterSet = new Set<number>();
    history.forEach(h => {
      if (h.clusterMetrics) {
        h.clusterMetrics.forEach(cm => clusterSet.add(cm.clusterId));
      }
    });
    return Array.from(clusterSet).sort((a, b) => a - b);
  }, [history]);

  // Cluster chart data - all clusters on the same chart
  const clusterChartData = useMemo(() => {
    return history.map((h) => {
      const dataPoint: Record<string, number> = { round: h.round + 1 };
      if (h.clusterMetrics) {
        h.clusterMetrics.forEach(cm => {
          dataPoint[`cluster_${cm.clusterId}`] = cm.accuracy * 100;
        });
      }
      return dataPoint;
    });
  }, [history]);

  // Get all unique client IDs across all rounds
  const availableClients = useMemo(() => {
    const clientSet = new Map<string, string>(); // id -> name
    history.forEach(h => {
      if (h.clientMetrics) {
        h.clientMetrics.forEach(cm => {
          if (!clientSet.has(cm.clientId)) {
            clientSet.set(cm.clientId, cm.clientName);
          }
        });
      }
    });
    return Array.from(clientSet.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [history]);

  // Client chart data - all clients on the same chart
  const clientChartData = useMemo(() => {
    return history.map((h) => {
      const dataPoint: Record<string, number> = { round: h.round + 1 };
      if (h.clientMetrics) {
        h.clientMetrics.forEach(cm => {
          dataPoint[`${cm.clientId}_loss`] = cm.loss;
          dataPoint[`${cm.clientId}_acc`] = cm.accuracy * 100;
          dataPoint[`${cm.clientId}_test`] = cm.testAccuracy * 100;
          if (cm.gradientNorm !== undefined) {
            dataPoint[`${cm.clientId}_grad`] = cm.gradientNorm;
          }
        });
      }
      return dataPoint;
    });
  }, [history]);

  const latestMetrics = history[history.length - 1];
  const previousMetrics = history[history.length - 2];

  const lossImproved = previousMetrics && latestMetrics && latestMetrics.globalLoss < previousMetrics.globalLoss;
  const accuracyImproved = previousMetrics && latestMetrics && latestMetrics.globalAccuracy > previousMetrics.globalAccuracy;

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Métriques Globales</CardTitle>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            <p>Lancez l'entraînement pour voir les métriques</p>
          </div>
        ) : (
          <Tabs defaultValue="server" className="w-full">
            <TabsList className="w-full grid grid-cols-4 bg-muted/30 mb-4">
              <TabsTrigger value="server" className="gap-2">
                <Server className="w-4 h-4" />
                Serveur
              </TabsTrigger>
              <TabsTrigger value="clusters" className="gap-2">
                <Layers className="w-4 h-4" />
                Clusters
              </TabsTrigger>
              <TabsTrigger value="clients" className="gap-2">
                <Users className="w-4 h-4" />
                Clients
              </TabsTrigger>
              <TabsTrigger value="3d" className="gap-2">
                <Box className="w-4 h-4" />
                3D
              </TabsTrigger>
            </TabsList>

            {/* Server Tab - Global model metrics */}
            <TabsContent value="server">
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-4 rounded-lg bg-muted/30 border border-border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Loss Globale</span>
                    {lossImproved && <TrendingDown className="w-4 h-4 text-success" />}
                  </div>
                  <p className="text-2xl font-mono font-bold text-foreground">
                    {latestMetrics?.globalLoss.toFixed(4) || '—'}
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-muted/30 border border-border">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Précision Globale</span>
                    {accuracyImproved && <TrendingUp className="w-4 h-4 text-success" />}
                  </div>
                  <p className="text-2xl font-mono font-bold text-foreground">
                    {latestMetrics ? `${(latestMetrics.globalAccuracy * 100).toFixed(1)}%` : '—'}
                  </p>
                </div>
              </div>

              <div className="h-[250px]">
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
                      yAxisId="left"
                      stroke="hsl(215, 20%, 55%)"
                      fontSize={12}
                      tickLine={false}
                      domain={[0, 'auto']}
                    />
                    <YAxis 
                      yAxisId="right"
                      orientation="right"
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
                      formatter={(value: number, name: string) => [
                        name === 'loss' ? value.toFixed(4) : `${value.toFixed(1)}%`,
                        name === 'loss' ? 'Loss' : 'Précision'
                      ]}
                    />
                    <Legend 
                      wrapperStyle={{ paddingTop: '10px' }}
                      formatter={(value) => value === 'loss' ? 'Loss' : 'Précision'}
                    />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="loss"
                      stroke="hsl(174, 72%, 56%)"
                      strokeWidth={2}
                      dot={{ fill: 'hsl(174, 72%, 56%)', strokeWidth: 0, r: 4 }}
                      activeDot={{ r: 6, fill: 'hsl(174, 72%, 56%)' }}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="accuracy"
                      stroke="hsl(142, 72%, 45%)"
                      strokeWidth={2}
                      dot={{ fill: 'hsl(142, 72%, 45%)', strokeWidth: 0, r: 4 }}
                      activeDot={{ r: 6, fill: 'hsl(142, 72%, 45%)' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              {/* Similarity matrix panel */}
              <SimilarityMatrix history={history} />
            </TabsContent>

            {/* Clusters Tab - All clusters metrics */}
            <TabsContent value="clusters">
              {availableClusters.length === 0 ? (
                <div className="h-[250px] flex items-center justify-center text-muted-foreground">
                  <p>Aucun cluster détecté — lancez un round d'entraînement</p>
                </div>
              ) : (
                <>
                  <div className="mb-4 p-3 rounded-lg bg-muted/30 border border-border">
                    <span className="text-sm text-muted-foreground">
                      {availableClusters.length} cluster{availableClusters.length > 1 ? 's' : ''} détecté{availableClusters.length > 1 ? 's' : ''}
                    </span>
                  </div>
                  
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={clusterChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
                            const clusterId = name.replace('cluster_', '');
                            return [`${value.toFixed(1)}%`, `Cluster #${parseInt(clusterId) + 1}`];
                          }}
                        />
                        <Legend 
                          wrapperStyle={{ paddingTop: '10px' }}
                          formatter={(value) => {
                            const clusterId = value.replace('cluster_', '');
                            return `Cluster #${parseInt(clusterId) + 1}`;
                          }}
                        />
                        {availableClusters.map((clusterId, idx) => (
                          <Line
                            key={clusterId}
                            type="monotone"
                            dataKey={`cluster_${clusterId}`}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={2}
                            dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 0, r: 3 }}
                            activeDot={{ r: 5, fill: COLORS[idx % COLORS.length] }}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </TabsContent>

            {/* Clients Tab - All clients metrics */}
            <TabsContent value="clients">
              {availableClients.length === 0 ? (
                <div className="h-[250px] flex items-center justify-center text-muted-foreground">
                  <p>Aucune métrique client — lancez un round d'entraînement</p>
                </div>
              ) : (
                <>
                  <div className="mb-4 p-3 rounded-lg bg-muted/30 border border-border">
                    <span className="text-sm text-muted-foreground">
                      {availableClients.length} client{availableClients.length > 1 ? 's' : ''} — Précision d'entraînement
                    </span>
                  </div>
                  
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={clientChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
                            const clientId = name.replace('_acc', '');
                            const clientName = availableClients.find(([id]) => id === clientId)?.[1] || clientId;
                            return [`${value.toFixed(1)}%`, clientName];
                          }}
                        />
                        <Legend 
                          wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }}
                          formatter={(value) => {
                            const clientId = value.replace('_acc', '');
                            return availableClients.find(([id]) => id === clientId)?.[1] || clientId;
                          }}
                        />
                        {availableClients.map(([clientId], idx) => (
                          <Line
                            key={clientId}
                            type="monotone"
                            dataKey={`${clientId}_acc`}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={2}
                            dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 0, r: 2 }}
                            activeDot={{ r: 4, fill: COLORS[idx % COLORS.length] }}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-6 p-3 rounded-lg bg-muted/30 border border-border">
                    <span className="text-sm text-muted-foreground">
                      Précision de test (évaluation sur données personnalisées)
                    </span>
                  </div>
                  
                  <div className="h-[250px] mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={clientChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
                            const clientId = name.replace('_test', '');
                            const clientName = availableClients.find(([id]) => id === clientId)?.[1] || clientId;
                            return [`${value.toFixed(1)}%`, clientName];
                          }}
                        />
                        <Legend 
                          wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }}
                          formatter={(value) => {
                            const clientId = value.replace('_test', '');
                            return availableClients.find(([id]) => id === clientId)?.[1] || clientId;
                          }}
                        />
                        {availableClients.map(([clientId], idx) => (
                          <Line
                            key={clientId}
                            type="monotone"
                            dataKey={`${clientId}_test`}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 0, r: 2 }}
                            activeDot={{ r: 4, fill: COLORS[idx % COLORS.length] }}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Gradient Norm Chart */}
                  <div className="mt-6 p-3 rounded-lg bg-muted/30 border border-border">
                    <span className="text-sm text-muted-foreground">
                      Norme des gradients (changement de poids après entraînement local)
                    </span>
                  </div>
                  
                  <div className="h-[250px] mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={clientChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
                            const clientId = name.replace('_grad', '');
                            const clientName = availableClients.find(([id]) => id === clientId)?.[1] || clientId;
                            return [value.toFixed(4), clientName];
                          }}
                        />
                        <Legend 
                          wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }}
                          formatter={(value) => {
                            const clientId = value.replace('_grad', '');
                            return availableClients.find(([id]) => id === clientId)?.[1] || clientId;
                          }}
                        />
                        {availableClients.map(([clientId], idx) => (
                          <Line
                            key={clientId}
                            type="monotone"
                            dataKey={`${clientId}_grad`}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={2}
                            dot={{ fill: COLORS[idx % COLORS.length], strokeWidth: 0, r: 2 }}
                            activeDot={{ r: 4, fill: COLORS[idx % COLORS.length] }}
                            connectNulls
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </TabsContent>

            {/* 3D Visualization Tab */}
            <TabsContent value="3d">
              <ModelVisualization3D 
                history={history}
                clientModels={clientModels}
                clusterModels={clusterModels}
                globalModel={globalModel}
                loadedVisualizations={loadedVisualizations}
              />
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
};