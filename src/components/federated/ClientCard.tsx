import { ClientState } from '@/lib/federated/types';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { 
  CircleDot, 
  Download, 
  Cpu, 
  Upload, 
  CheckCircle2, 
  AlertCircle,
  Database
} from 'lucide-react';

interface ClientCardProps {
  client: ClientState;
}

const statusConfig = {
  idle: {
    icon: CircleDot,
    label: 'En attente',
    color: 'text-muted-foreground',
    bgColor: 'bg-muted/30',
    borderColor: 'border-border',
  },
  receiving: {
    icon: Download,
    label: 'Réception',
    color: 'text-info',
    bgColor: 'bg-info/10',
    borderColor: 'border-info/30',
  },
  training: {
    icon: Cpu,
    label: 'Entraînement',
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
  },
  sending: {
    icon: Upload,
    label: 'Envoi',
    color: 'text-primary',
    bgColor: 'bg-primary/10',
    borderColor: 'border-primary/30',
  },
  completed: {
    icon: CheckCircle2,
    label: 'Terminé',
    color: 'text-success',
    bgColor: 'bg-success/10',
    borderColor: 'border-success/30',
  },
  error: {
    icon: AlertCircle,
    label: 'Erreur',
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
    borderColor: 'border-destructive/30',
  },
  evaluating: {
    icon: Cpu,
    label: 'Evaluation',
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
    borderColor: 'border-destructive/30',
  },
};

export const ClientCard = ({ client }: ClientCardProps) => {
  const config = statusConfig[client.status];
  const StatusIcon = config.icon;

  return (
    <Card 
      className={cn(
        'transition-all duration-300 border',
        config.bgColor,
        config.borderColor,
        client.status === 'training' && 'animate-pulse-glow'
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h4 className="font-medium text-sm text-foreground">{client.name}</h4>
            <p className="text-xs font-mono text-muted-foreground">{client.id}</p>
          </div>
          <div className={cn('p-1.5 rounded-md', config.bgColor)}>
            <StatusIcon className={cn('w-4 h-4', config.color)} />
          </div>
        </div>

        <div className="space-y-3">
          {/* Status */}
          <div className="flex items-center justify-between">
            <span className={cn('text-xs font-medium', config.color)}>
              {config.label}
            </span>
            {client.status !== 'idle' && (
              <span className="text-xs font-mono text-muted-foreground">
                {Math.round(client.progress)}%
              </span>
            )}
          </div>

          {/* Progress Bar */}
          {client.status !== 'idle' && (
            <Progress 
              value={client.progress} 
              className={cn(
                'h-1.5',
                client.status === 'completed' && '[&>div]:bg-success',
                client.status === 'training' && '[&>div]:bg-warning',
                client.status === 'receiving' && '[&>div]:bg-info',
                client.status === 'sending' && '[&>div]:bg-primary',
              )}
            />
          )}

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-border/50">
            <div className="text-center">
              <Database className="w-3 h-3 mx-auto mb-1 text-muted-foreground" />
              <p className="text-xs font-mono text-muted-foreground">
                {(client.dataSize / 1000).toFixed(1)}k
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-0.5">Loss</p>
              <p className="text-xs font-mono text-foreground">
                {client.localLoss > 0 ? client.localLoss.toFixed(3) : '—'}
              </p>
            </div>
          </div>

          {/* Test Accuracy (prominent display) */}
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-border/50">
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-0.5">Train Acc</p>
              <p className="text-xs font-mono text-foreground">
                {client.localAccuracy > 0 ? `${(client.localAccuracy * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-0.5">Test Acc</p>
              <p className={cn(
                'text-xs font-mono font-semibold',
                client.localTestAccuracy > 0 ? 'text-success' : 'text-foreground'
              )}>
                {client.localTestAccuracy > 0 ? `${(client.localTestAccuracy * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
          </div>

          {/* Rounds Participated */}
          {client.roundsParticipated > 0 && (
            <p className="text-xs text-center text-muted-foreground">
              {client.roundsParticipated} round{client.roundsParticipated > 1 ? 's' : ''} participé{client.roundsParticipated > 1 ? 's' : ''}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
