import { RoundMetrics } from '@/lib/federated/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingDown, TrendingUp } from 'lucide-react';

interface MetricsChartProps {
  history: RoundMetrics[];
}

export const MetricsChart = ({ history }: MetricsChartProps) => {
  const chartData = history.map((h) => ({
    round: h.round + 1,
    loss: h.globalLoss,
    accuracy: h.globalAccuracy * 100,
  }));

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
          <>
            {/* Summary Stats */}
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

            {/* Chart */}
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
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
          </>
        )}
      </CardContent>
    </Card>
  );
};
