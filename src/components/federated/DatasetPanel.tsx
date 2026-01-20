import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Database, BarChart3, HelpCircle } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export type DatasetType = 'mnist';
export type DistributionType = '70-30';

interface DatasetPanelProps {
  dataset: DatasetType;
  distribution: DistributionType;
  onDatasetChange: (dataset: DatasetType) => void;
  onDistributionChange: (distribution: DistributionType) => void;
  disabled?: boolean;
}

const datasetOptions: { value: DatasetType; label: string; description: string }[] = [
  { value: 'mnist', label: 'MNIST', description: 'Chiffres manuscrits (0-9)' },
];

const distributionOptions: { value: DistributionType; label: string; description: string }[] = [
  { value: '70-30', label: '70/30', description: '70% classe principale, 30% autres' },
];

const DatasetPanel: React.FC<DatasetPanelProps> = ({
  dataset,
  distribution,
  onDatasetChange,
  onDistributionChange,
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
                    <p>Stratégie de répartition des données entre clients. 70/30 signifie que chaque client a 70% de sa classe principale et 30% des autres classes.</p>
                  </TooltipContent>
                </Tooltip>
              </Label>
            <Select
              value={distribution}
              onValueChange={(value) => onDistributionChange(value as DistributionType)}
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
        </div>
      </CardContent>
    </Card>
  </TooltipProvider>
  );
};

export default DatasetPanel;
