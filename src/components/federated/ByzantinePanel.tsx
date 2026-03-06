import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ShieldAlert, HelpCircle, Plus, Trash2 } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export type ByzantineAttack = 'local-model-poisoning' | 'label-flipping' | 'gradient-scaling' | 'hdpa';
export interface ByzantineInterval { start: number; end: number; }

interface ByzantinePanelProps {
  byzantineCount: number;
  attackMethod: ByzantineAttack;
  activeIntervals: ByzantineInterval[];
  onByzantineCountChange: (count: number) => void;
  onAttackMethodChange: (method: ByzantineAttack) => void;
  onActiveIntervalsChange: (intervals: ByzantineInterval[]) => void;
  disabled?: boolean;
  maxClients?: number;
}

const attackOptions: { value: ByzantineAttack; label: string; description: string }[] = [
  { value: 'local-model-poisoning', label: 'Local Model Poisoning', description: 'Le client envoie un modèle empoisonné au serveur' },
  { value: 'hdpa', label: 'HDPA', description: 'Empoisonnement via encodage hyperdimensionnel des données' },
  { value: 'label-flipping', label: 'Label Flipping', description: 'Les labels d\'entraînement sont inversés' },
  { value: 'gradient-scaling', label: 'Gradient Scaling', description: 'Les gradients sont multipliés par un facteur malveillant' },
];

const ByzantinePanel: React.FC<ByzantinePanelProps> = ({
  byzantineCount,
  attackMethod,
  activeIntervals,
  onByzantineCountChange,
  onAttackMethodChange,
  onActiveIntervalsChange,
  disabled = false,
  maxClients = 6,
}) => {
  const max = Math.min(maxClients, 6);

  const addInterval = () => {
    const lastEnd = activeIntervals.length > 0 ? activeIntervals[activeIntervals.length - 1].end + 1 : 0;
    onActiveIntervalsChange([...activeIntervals, { start: lastEnd, end: lastEnd + 5 }]);
  };

  const removeInterval = (index: number) => {
    onActiveIntervalsChange(activeIntervals.filter((_, i) => i !== index));
  };

  const updateInterval = (index: number, field: 'start' | 'end', value: number) => {
    const updated = activeIntervals.map((iv, i) =>
      i === index ? { ...iv, [field]: Math.max(0, value) } : iv
    );
    onActiveIntervalsChange(updated);
  };

  return (
    <TooltipProvider>
      <Card className="w-full border-border/50 bg-card/50">
        <CardHeader className="py-3">
          <CardTitle className="text-base flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-destructive" />
            Byzantin
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4 space-y-4">
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

          {/* Active intervals */}
          {byzantineCount > 0 && (
            <div className="space-y-2">
              <Label className="text-sm font-medium flex items-center gap-1.5">
                Périodes d'activité
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs bg-popover border border-border">
                    <p>Intervalles de rounds pendant lesquels les clients byzantins sont actifs. Si aucun intervalle, ils sont actifs à chaque round.</p>
                  </TooltipContent>
                </Tooltip>
              </Label>

              <div className="space-y-2">
                {activeIntervals.map((iv, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground w-12">Début</span>
                    <Input
                      type="number"
                      min={0}
                      value={iv.start}
                      onChange={(e) => updateInterval(index, 'start', parseInt(e.target.value) || 0)}
                      disabled={disabled}
                      className="w-20 h-8 text-sm"
                    />
                    <span className="text-xs text-muted-foreground w-8">Fin</span>
                    <Input
                      type="number"
                      min={0}
                      value={iv.end}
                      onChange={(e) => updateInterval(index, 'end', parseInt(e.target.value) || 0)}
                      disabled={disabled}
                      className="w-20 h-8 text-sm"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive"
                      onClick={() => removeInterval(index)}
                      disabled={disabled}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                ))}

                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs gap-1"
                  onClick={addInterval}
                  disabled={disabled}
                >
                  <Plus className="w-3.5 h-3.5" />
                  Ajouter un intervalle
                </Button>

                {activeIntervals.length === 0 && (
                  <p className="text-xs text-muted-foreground italic">
                    Aucun intervalle défini — actifs à chaque round.
                  </p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

export default ByzantinePanel;
