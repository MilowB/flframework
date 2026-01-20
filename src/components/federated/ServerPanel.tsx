import { Input } from '@/components/ui/input';
import { ServerConfig } from '@/lib/federated/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { aggregationMethods } from '@/lib/federated/aggregations';
import { Server, Layers, Users, Hash, HelpCircle } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface ServerPanelProps {
  config: ServerConfig;
  onConfigChange: (config: Partial<ServerConfig>) => void;
  disabled?: boolean;
  globalModelVersion: number;
}

const architectures = [
  { value: 'mlp-small', label: 'MLP Small (784→128→10)' },
];

export const ServerPanel = ({ config, onConfigChange, disabled, globalModelVersion }: ServerPanelProps) => {
  return (
    <TooltipProvider>
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
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Nombre total de cycles d'entraînement fédéré. Chaque round inclut l'envoi du modèle, l'entraînement local et l'agrégation.</p>
                  </TooltipContent>
                </Tooltip>
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
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Nombre total de clients participant à l'apprentissage fédéré. Chaque client possède ses propres données locales.</p>
                  </TooltipContent>
                </Tooltip>
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
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Nombre de clients sélectionnés aléatoirement à chaque round pour participer à l'entraînement.</p>
                  </TooltipContent>
                </Tooltip>
              </span>
              <span className="font-mono text-primary">{config.clientsPerRound}</span>
            </Label>
            <Slider
              value={[config.clientsPerRound]}
              onValueChange={([value]) => onConfigChange({ clientsPerRound: value, minClientsRequired: value })}
              min={1}
              max={10}
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
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs bg-popover border border-border">
                  <p>Structure du réseau de neurones. MLP Small : 784 entrées → 128 neurones cachés → 10 sorties.</p>
                </TooltipContent>
              </Tooltip>
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
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs bg-popover border border-border">
                  <p>Algorithme utilisé par le serveur pour combiner les modèles des clients. FedAvg pondère par la taille des données.</p>
                </TooltipContent>
              </Tooltip>
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
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs bg-popover border border-border">
                  <p>Stratégie pour assigner un modèle de cluster à chaque client. 1NN : plus proche voisin. Probabiliste : pondéré par distance.</p>
                </TooltipContent>
              </Tooltip>
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

          {/* Méthode de clustering */}
          <div className="space-y-2">
            <Label className="flex items-center gap-2 text-sm text-muted-foreground">
              Méthode de clustering
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs bg-popover border border-border">
                  <p>Algorithme de regroupement des clients similaires. Louvain/Leiden : détection de communautés. K-means : clustering classique.</p>
                </TooltipContent>
              </Tooltip>
            </Label>
            <Select
              value={config.clusteringMethod ?? 'louvain'}
              onValueChange={(value: 'louvain' | 'kmeans' | 'leiden') => onConfigChange({ clusteringMethod: value })}
              disabled={disabled}
            >
              <SelectTrigger className="bg-muted/50 border-border">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="louvain">
                  <span className="font-medium">Louvain</span>
                </SelectItem>
                <SelectItem value="kmeans">
                  <span className="font-medium">K-means</span>
                </SelectItem>
                <SelectItem value="leiden">
                  <span className="font-medium">Leiden</span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Matrice d'agreement */}
          <div className="flex items-center space-x-3 pt-2">
            <Checkbox
              id="agreement-matrix"
              checked={config.useAgreementMatrix ?? false}
              onCheckedChange={(checked) => onConfigChange({ useAgreementMatrix: checked === true })}
              disabled={disabled}
            />
            <Label 
              htmlFor="agreement-matrix" 
              className="text-sm text-muted-foreground cursor-pointer flex items-center gap-2"
            >
              Matrice d'agreement
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs bg-popover border border-border">
                  <p>Active le clustering par consensus. Exécute plusieurs runs de clustering et compte la fréquence de co-clustering entre clients.</p>
                </TooltipContent>
              </Tooltip>
            </Label>
          </div>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};