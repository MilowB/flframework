import { Input } from '@/components/ui/input';
import { ServerConfig } from '@/lib/federated/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { aggregationMethods } from '@/lib/federated/aggregations';
import { Server, Layers, Users, Hash } from 'lucide-react';

interface ServerPanelProps {
  config: ServerConfig;
  onConfigChange: (config: Partial<ServerConfig>) => void;
  disabled?: boolean;
  globalModelVersion: number;
}

const architectures = [
  { value: 'mlp-small', label: 'MLP Small (784→128→10)' },
  { value: 'mlp-medium', label: 'MLP Medium (784→256→128→10)' },
  { value: 'mlp-large', label: 'MLP Large (784→512→256→128→10)' },
  { value: 'cnn-simple', label: 'CNN Simple (32→64→128→10)' },
  { value: 'resnet-mini', label: 'ResNet Mini (64→128→256→512→10)' },
];

export const ServerPanel = ({ config, onConfigChange, disabled, globalModelVersion }: ServerPanelProps) => {
  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="p-2 rounded-lg bg-primary/10">
            <Server className="w-5 h-5 text-primary" />
          </div>
          <span>Serveur Central</span>
          <span className="ml-auto text-xs font-mono text-muted-foreground">
            v{globalModelVersion}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Nombre de rounds */}
        <div className="space-y-3">
          <Label className="flex items-center justify-between text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <Hash className="w-4 h-4" />
              Nombre de rounds
            </span>
            <span className="font-mono text-primary">{config.totalRounds}</span>
          </Label>
          <Slider
            value={[config.totalRounds]}
            onValueChange={([value]) => onConfigChange({ totalRounds: value })}
            min={1}
            max={50}
            step={1}
            disabled={disabled}
            className="[&_[role=slider]]:bg-primary"
          />
        </div>

        {/* Nombre de clients */}
        <div className="space-y-3">
          <Label className="flex items-center justify-between text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              Nombre de clients
            </span>
            <span className="font-mono text-primary">{config.clientCount ?? 6}</span>
          </Label>
          <Slider
            value={[config.clientCount ?? 6]}
            onValueChange={([value]) => onConfigChange({ clientCount: value })}
            min={2}
            max={10}
            step={1}
            disabled={disabled}
            className="[&_[role=slider]]:bg-primary"
          />
        </div>

        {/* Clients par round */}
        <div className="space-y-3">
          <Label className="flex items-center justify-between text-sm text-muted-foreground">
            <span className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              Clients par round
            </span>
            <span className="font-mono text-primary">{config.clientsPerRound}</span>
          </Label>
          <Slider
            value={[config.clientsPerRound]}
            onValueChange={([value]) => onConfigChange({ clientsPerRound: value })}
            min={1}
            max={10}
            step={1}
            disabled={disabled}
            className="[&_[role=slider]]:bg-primary"
          />
        </div>

        {/* Clients minimum requis */}
        <div className="space-y-3">
          <Label className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Clients minimum requis</span>
            <span className="font-mono text-primary">{config.minClientsRequired}</span>
          </Label>
          <Slider
            value={[config.minClientsRequired]}
            onValueChange={([value]) => onConfigChange({ minClientsRequired: value })}
            min={1}
            max={config.clientsPerRound}
            step={1}
            disabled={disabled}
            className="[&_[role=slider]]:bg-primary"
          />
        </div>

        {/* Architecture du modèle */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm text-muted-foreground">
            <Layers className="w-4 h-4" />
            Architecture du modèle
          </Label>
          <Select
            value={config.modelArchitecture}
            onValueChange={(value) => onConfigChange({ modelArchitecture: value })}
            disabled={disabled}
          >
            <SelectTrigger className="bg-muted/50 border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {architectures.map((arch) => (
                <SelectItem key={arch.value} value={arch.value}>
                  <span className="font-mono text-sm">{arch.label}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Méthode d'agrégation serveur */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm text-muted-foreground">
            Méthode d'agrégation serveur
          </Label>
          <Select
            value={config.aggregationMethod}
            onValueChange={(value: ServerConfig['aggregationMethod']) =>
              onConfigChange({ aggregationMethod: value })
            }
            disabled={disabled}
          >
            <SelectTrigger className="bg-muted/50 border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(aggregationMethods).map(([key, method]) => (
                <SelectItem key={key} value={key}>
                  <div className="flex flex-col">
                    <span className="font-medium">{method.name}</span>
                    <span className="text-xs text-muted-foreground">{method.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Méthode d'affectation de modèle */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm text-muted-foreground">
            Méthode d'affectation de modèle
          </Label>
          <Select
            value={config.modelAssignmentMethod ?? '1NN'}
            onValueChange={(value: '1NN' | 'Probabiliste') => onConfigChange({ modelAssignmentMethod: value })}
            disabled={disabled}
          >
            <SelectTrigger className="bg-muted/50 border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1NN">
                <span className="font-medium">1NN</span>
              </SelectItem>
              <SelectItem value="Probabiliste">
                <span className="font-medium">Probabiliste</span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
};