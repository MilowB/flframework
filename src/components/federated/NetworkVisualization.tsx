import { ClientState } from '@/lib/federated/types';
import { cn } from '@/lib/utils';
import { Server, ArrowUp, ArrowDown } from 'lucide-react';

interface NetworkVisualizationProps {
  clients: ClientState[];
  globalModelVersion: number;
}

export const NetworkVisualization = ({ clients, globalModelVersion }: NetworkVisualizationProps) => {
  const activeClients = clients.filter(c => c.status !== 'idle');
  
  return (
    <div className="relative p-8 rounded-xl bg-gradient-card border border-border shadow-card overflow-hidden">
      {/* Background Grid */}
      <div className="absolute inset-0 grid-pattern opacity-30" />
      
      {/* Glow Effect */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-full bg-primary/20 blur-3xl" />
      
      <div className="relative">
        {/* Server Node */}
        <div className="flex flex-col items-center mb-8">
          <div className={cn(
            'relative p-6 rounded-2xl border-2 transition-all duration-500',
            activeClients.length > 0 
              ? 'bg-primary/20 border-primary shadow-glow' 
              : 'bg-muted/30 border-border'
          )}>
            <Server className="w-10 h-10 text-primary" />
            <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-background border border-border">
              <span className="text-xs font-mono">v{globalModelVersion}</span>
            </div>
          </div>
          <h3 className="mt-4 text-sm font-medium text-foreground">Serveur Central</h3>
        </div>

        {/* Connection Lines & Client Nodes */}
        <div className="flex flex-wrap justify-center gap-4">
          {clients.map((client, index) => {
            const isActive = client.status !== 'idle';
            const isReceiving = client.status === 'receiving';
            const isSending = client.status === 'sending';
            
            return (
              <div 
                key={client.id}
                className="relative flex flex-col items-center"
              >
                {/* Connection Line */}
                <div className={cn(
                  'w-0.5 h-8 transition-all duration-300',
                  isActive ? 'bg-gradient-to-b from-primary to-transparent' : 'bg-border'
                )}>
                  {/* Data Flow Animation */}
                  {isReceiving && (
                    <div className="absolute top-0 left-1/2 -translate-x-1/2">
                      <ArrowDown className="w-4 h-4 text-info animate-bounce" />
                    </div>
                  )}
                  {isSending && (
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2">
                      <ArrowUp className="w-4 h-4 text-primary animate-bounce" />
                    </div>
                  )}
                </div>

                {/* Client Node */}
                <div 
                  className={cn(
                    'relative p-3 rounded-xl border transition-all duration-300',
                    client.status === 'idle' && 'bg-muted/20 border-border',
                    client.status === 'receiving' && 'bg-info/10 border-info/50',
                    client.status === 'training' && 'bg-warning/10 border-warning/50 animate-pulse',
                    client.status === 'sending' && 'bg-primary/10 border-primary/50',
                    client.status === 'completed' && 'bg-success/10 border-success/50',
                    client.status === 'error' && 'bg-destructive/10 border-destructive/50',
                  )}
                >
                  <div className={cn(
                    'w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold',
                    isActive ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
                  )}>
                    {index + 1}
                  </div>
                  
                  {/* Status Indicator */}
                  <div className={cn(
                    'absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-background',
                    client.status === 'idle' && 'bg-muted-foreground',
                    client.status === 'receiving' && 'bg-info',
                    client.status === 'training' && 'bg-warning animate-pulse',
                    client.status === 'sending' && 'bg-primary',
                    client.status === 'completed' && 'bg-success',
                    client.status === 'error' && 'bg-destructive',
                  )} />
                </div>

                <span className="mt-2 text-xs text-muted-foreground text-center max-w-[80px] truncate">
                  {client.name.split(' ')[0]}
                </span>
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 mt-8 pt-6 border-t border-border/50">
          {[
            { status: 'idle', label: 'En attente', color: 'bg-muted-foreground' },
            { status: 'receiving', label: 'Réception', color: 'bg-info' },
            { status: 'training', label: 'Entraînement', color: 'bg-warning' },
            { status: 'sending', label: 'Envoi', color: 'bg-primary' },
            { status: 'completed', label: 'Terminé', color: 'bg-success' },
          ].map((item) => (
            <div key={item.status} className="flex items-center gap-2">
              <div className={cn('w-2.5 h-2.5 rounded-full', item.color)} />
              <span className="text-xs text-muted-foreground">{item.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
