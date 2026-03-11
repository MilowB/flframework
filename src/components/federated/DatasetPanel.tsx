import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Database, BarChart3, HelpCircle } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Slider } from '@/components/ui/slider';

export type DatasetType = 'mnist';
export type DistributionType = '70-30' | 'dirichlet' | 'iid';

interface DatasetPanelProps {
  dataset: DatasetType;
  distribution: DistributionType;
  dirichletAlpha: number;
  muFraction: number;
  onDatasetChange: (dataset: DatasetType) => void;
  onDistributionChange: (distribution: DistributionType) => void;
  onDirichletAlphaChange: (alpha: number) => void;
  onMuFractionChange: (mu: number) => void;
  disabled?: boolean;
}

const datasetOptions: { value: DatasetType; label: string; description: string }[] = [
  { value: 'mnist', label: 'MNIST', description: 'Chiffres manuscrits (0-9) - utilise un subset de 600 données / client pour l\'entrainement.'},
];

const distributionOptions: { value: DistributionType; label: string; description: string }[] = [
  { value: '70-30', label: '𝜇-Fraction', description: '𝜇% classe principale, reste réparti' },
  { value: 'iid', label: 'IID', description: 'N échantillons aléatoires par client' },
  { value: 'dirichlet', label: 'Dirichlet', description: 'Distribution non-IID contrôlée par α' },
];

const DatasetPanel: React.FC<DatasetPanelProps> = ({
  dataset,
  distribution,
  dirichletAlpha,
  muFraction,
  onDatasetChange,
  onDistributionChange,
  onDirichletAlphaChange,
  onMuFractionChange,
  disabled = false,
}) => {
  return (
    <TooltipProvider>
      <Card className="w-full border-border/50 bg-card/50">
        <CardHeader className="py-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Database className="w-4 h-4 text-primary" />
            Données
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Dataset Selection */}
            <div className="space-y-2">
              <Label className="text-sm font-medium flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5" />
                Jeu de données
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Dataset utilisé pour l'entraînement. MNIST contient 60 000 images de chiffres manuscrits (0-9) de 28×28 pixels.</p>
                  </TooltipContent>
                </Tooltip>
              </Label>
              <Select
                value={dataset}
                onValueChange={(value) => onDatasetChange(value as DatasetType)}
                disabled={disabled}
              >
                <SelectTrigger className="w-full bg-background">
                  <SelectValue placeholder="Sélectionner un dataset" />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {datasetOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      <div className="flex flex-col">
                        <span className="font-medium">{option.label}</span>
                        <span className="text-xs text-muted-foreground">{option.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Distribution Selection */}
            <div className="space-y-2">
              <Label className="text-sm font-medium flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5" />
                Distribution
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Stratégie de répartition des données entre clients. IID : N échantillons aléatoires par client. 40/60 : 40% classe principale, 60% autres. Dirichlet : distribution contrôlée par α (petit α = très non-IID, grand α ≈ IID).</p>
                  </TooltipContent>
                </Tooltip>
              </Label>
              <Select
                value={distribution}
                onValueChange={(value) => {
                  console.log(`[DatasetPanel] onDistributionChange called with value=${value}`);
                  onDistributionChange(value as DistributionType);
                }}
                disabled={disabled}
              >
                <SelectTrigger className="w-full bg-background">
                  <SelectValue placeholder="Sélectionner une distribution" />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {distributionOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      <div className="flex flex-col">
                        <span className="font-medium">{option.label}</span>
                        <span className="text-xs text-muted-foreground">{option.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Mu Fraction - shown only when 𝜇-Fraction is selected */}
            {distribution === '70-30' && (
              <div className="md:col-span-2 space-y-2 p-3 rounded-md border border-border/50 bg-muted/20">
                <Label className="text-sm font-medium flex items-center gap-1.5">
                  𝜇 (%)
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs bg-popover border border-border">
                      <p>Pourcentage de la classe majoritaire dans le dataset local de chaque client. Les {100 - muFraction}% restants sont répartis uniformément entre les autres classes.</p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground whitespace-nowrap">10%</span>
                  <Slider
                    value={[muFraction]}
                    onValueChange={([v]) => onMuFractionChange(v)}
                    min={10}
                    max={90}
                    step={1}
                    disabled={disabled}
                    className="flex-1"
                  />
                  <span className="text-xs text-muted-foreground whitespace-nowrap">90%</span>
                  <Input
                    type="number"
                    value={muFraction}
                    onChange={(e) => {
                      const v = parseInt(e.target.value);
                      if (!isNaN(v) && v >= 10 && v <= 90) onMuFractionChange(v);
                    }}
                    min={10}
                    max={90}
                    step={1}
                    disabled={disabled}
                    className="w-20 h-8 text-sm bg-background"
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {muFraction}% classe majoritaire / {100 - muFraction}% autres classes
                </p>
              </div>
            )}

            {/* Dirichlet Alpha - shown only when Dirichlet is selected */}
            {distribution === 'dirichlet' && (
              <div className="md:col-span-2 space-y-2 p-3 rounded-md border border-border/50 bg-muted/20">
                <Label className="text-sm font-medium flex items-center gap-1.5">
                  α (Alpha)
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs bg-popover border border-border">
                      <p>Paramètre de concentration de la distribution de Dirichlet. Plus α est petit (→0), plus la distribution est non-IID (chaque client se spécialise sur peu de classes). Plus α est grand (→∞), plus la distribution tend vers l'IID uniforme.</p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted-foreground whitespace-nowrap">Non-IID</span>
                  <Slider
                    value={[dirichletAlpha]}
                    onValueChange={([v]) => onDirichletAlphaChange(v)}
                    min={0.01}
                    max={10}
                    step={0.01}
                    disabled={disabled}
                    className="flex-1"
                  />
                  <span className="text-xs text-muted-foreground whitespace-nowrap">IID</span>
                  <Input
                    type="number"
                    value={dirichletAlpha}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (!isNaN(v) && v > 0) onDirichletAlphaChange(v);
                    }}
                    min={0.01}
                    step={0.01}
                    disabled={disabled}
                    className="w-20 h-8 text-sm bg-background"
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  α = {dirichletAlpha < 0.1 ? 'très non-IID' : dirichletAlpha < 1 ? 'non-IID' : dirichletAlpha < 5 ? 'modérément IID' : 'quasi-IID'}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

export default DatasetPanel;
