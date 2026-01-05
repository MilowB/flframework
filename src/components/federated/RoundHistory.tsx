import { RoundMetrics } from '@/lib/federated/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Clock, Users, Zap } from 'lucide-react';

interface RoundHistoryProps {
  history: RoundMetrics[];
}

export const RoundHistory = ({ history }: RoundHistoryProps) => {
  const reversedHistory = [...history].reverse();

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Clock className="w-5 h-5 text-primary" />
          Historique des Rounds
        </CardTitle>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <div className="h-[200px] flex items-center justify-center text-muted-foreground">
            <p className="text-sm">Aucun round complété</p>
          </div>
        ) : (
          <ScrollArea className="h-[300px] pr-4">
            <div className="space-y-3">
              {reversedHistory.map((round) => (
                <div 
                  key={round.round}
                  className="p-4 rounded-lg bg-muted/20 border border-border hover:border-primary/30 transition-colors"
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-foreground">
                      Round {round.round + 1}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(round.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div>
                      <p className="text-muted-foreground text-xs mb-1">Loss</p>
                      <p className="font-mono text-primary">{round.globalLoss.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground text-xs mb-1">Précision</p>
                      <p className="font-mono text-success">{(round.globalAccuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground text-xs mb-1 flex items-center gap-1">
                        <Zap className="w-3 h-3" />
                        Agrég.
                      </p>
                      <p className="font-mono">{round.aggregationTime}ms</p>
                    </div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <div className="flex items-center gap-2">
                      <Users className="w-3 h-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        {round.participatingClients.length} clients participants
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
};
