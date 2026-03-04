import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ShieldAlert, HelpCircle } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export type ByzantineAttack = 'local-model-poisoning' | 'label-flipping' | 'gradient-scaling';

interface ByzantinePanelProps {
  byzantineCount: number;
  attackMethod: ByzantineAttack;
  onByzantineCountChange: (count: number) => void;
  onAttackMethodChange: (method: ByzantineAttack) => void;
  disabled?: boolean;
  maxClients?: number;
}

const attackOptions: { value: ByzantineAttack; label: string; description: string }[] = [
  { value: 'local-model-poisoning', label: 'Local Model Poisoning', description: 'Le client envoie un modèle empoisonné au serveur' },
  { value: 'label-flipping', label: 'Label Flipping', description: 'Les labels d\'entraînement sont inversés' },
  { value: 'gradient-scaling', label: 'Gradient Scaling', description: 'Les gradients sont multipliés par un facteur malveillant' },
];

const ByzantinePanel: React.FC<ByzantinePanelProps> = ({
  byzantineCount,
  attackMethod,
  onByzantineCountChange,
  onAttackMethodChange,
  disabled = false,
  maxClients = 6,
}) => {
  const max = Math.min(maxClients, 6);

  return (
    <TooltipProvider>
      <Card className="w-full border-border/50 bg-card/50">
        <CardHeader className="py-3">
          <CardTitle className="text-base flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-destructive" />
            Byzantin
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Byzantine count */}
            <div className="space-y-2">
              <Label className="text-sm font-medium flex items-center gap-1.5">
                Clients byzantins
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Nombre de clients qui se comportent de manière malveillante pendant l'entraînement fédéré.</p>
                  </TooltipContent>
                </Tooltip>
              </Label>
              <div className="flex items-center gap-3">
                <Slider
                  value={[byzantineCount]}
                  onValueChange={([v]) => onByzantineCountChange(v)}
                  min={0}
                  max={max}
                  step={1}
                  disabled={disabled}
                  className="flex-1"
                />
                <span className="text-sm font-mono w-6 text-center text-foreground">{byzantineCount}</span>
              </div>
            </div>

            {/* Attack method */}
            <div className="space-y-2">
              <Label className="text-sm font-medium flex items-center gap-1.5">
                Méthode d'attaque
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Type d'attaque byzantine utilisée par les clients malveillants.</p>
                  </TooltipContent>
                </Tooltip>
              </Label>
              <Select
                value={attackMethod}
                onValueChange={(v) => onAttackMethodChange(v as ByzantineAttack)}
                disabled={disabled || byzantineCount === 0}
              >
                <SelectTrigger className="w-full bg-background">
                  <SelectValue placeholder="Méthode d'attaque" />
                </SelectTrigger>
                <SelectContent className="bg-popover z-50">
                  {attackOptions.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      <div className="flex flex-col">
                        <span className="font-medium">{opt.label}</span>
                        <span className="text-xs text-muted-foreground">{opt.description}</span>
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

export default ByzantinePanel;
