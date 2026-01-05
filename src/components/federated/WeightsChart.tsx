import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { RoundMetrics } from '@/lib/federated/types';
import { Layers } from 'lucide-react';

interface WeightsChartProps {
  history: RoundMetrics[];
}

export const WeightsChart = ({ history }: WeightsChartProps) => {
  const data = history
    .filter(h => h.weightsSnapshot)
    .map(h => ({
      round: h.round + 1,
      'W1 Mean': h.weightsSnapshot!.W1Mean,
      'W1 Std': h.weightsSnapshot!.W1Std,
      'W2 Mean': h.weightsSnapshot!.W2Mean,
      'W2 Std': h.weightsSnapshot!.W2Std,
      'b1 Mean': h.weightsSnapshot!.b1Mean,
      'b2 Mean': h.weightsSnapshot!.b2Mean,
    }));

  if (data.length === 0) {
    return (
      <Card className="bg-card/50 border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" />
            Évolution des Poids
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px] flex items-center justify-center text-muted-foreground text-sm">
            Les poids apparaîtront après le premier round
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-card/50 border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" />
          Évolution des Poids du MLP
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis 
                dataKey="round" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                tickLine={false}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
                formatter={(value: number) => value.toFixed(4)}
              />
              <Legend 
                wrapperStyle={{ fontSize: '10px' }}
                iconSize={8}
              />
              <Line
                type="monotone"
                dataKey="W1 Mean"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="W1 Std"
                stroke="hsl(var(--primary))"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="W2 Mean"
                stroke="hsl(var(--accent))"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="W2 Std"
                stroke="hsl(var(--accent))"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="b1 Mean"
                stroke="hsl(142, 76%, 50%)"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="b2 Mean"
                stroke="hsl(45, 93%, 58%)"
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};
