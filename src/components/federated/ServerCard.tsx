import { ServerStatus } from '@/lib/federated/types';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { 
  Server,
  Upload, 
  Clock, 
  Download, 
  Activity,
  CheckCircle2 
} from 'lucide-react';

interface ServerCardProps {
  status: ServerStatus;
  modelVersion: number;
}

const statusConfig = {
  idle: {
    icon: Clock,
    label: 'En attente',
    color: 'text-muted-foreground',
    bgColor: 'bg-muted/30',
    borderColor: 'border-border',
  },
  sending: {
    icon: Upload,
    label: 'Envoi',
    color: 'text-primary',
    bgColor: 'bg-primary/10',
    borderColor: 'border-primary/30',
  },
  waiting: {
    icon: Clock,
    label: 'En attente',
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
  },
  receiving: {
    icon: Download,
    label: 'Réception',
    color: 'text-info',
    bgColor: 'bg-info/10',
    borderColor: 'border-info/30',
  },
  evaluating: {
    icon: Activity,
    label: 'Évaluation',
    color: 'text-accent',
    bgColor: 'bg-accent/10',
    borderColor: 'border-accent/30',
  },
  completed: {
    icon: CheckCircle2,
    label: 'Terminé',
    color: 'text-success',
    bgColor: 'bg-success/10',
    borderColor: 'border-success/30',
  },
};

export const ServerCard = ({ status, modelVersion }: ServerCardProps) => {
  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <Card 
      className={cn(
        'transition-all duration-300 border-2',
        config.bgColor,
        config.borderColor,
        status === 'evaluating' && 'animate-pulse-glow'
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h4 className="font-medium text-sm text-foreground flex items-center gap-2">
              <Server className="w-4 h-4 text-primary" />
              Serveur Central
            </h4>
            <p className="text-xs font-mono text-muted-foreground">global-server</p>
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
          </div>

          {/* Model Version */}
          <div className="pt-2 border-t border-border/50">
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-0.5">Version modèle</p>
              <p className="text-lg font-mono font-bold text-foreground">
                v{modelVersion}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
