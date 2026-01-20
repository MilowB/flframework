import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';

interface GravityHyperparams {
  gravitationConstant: number;
  clusterWeight: number;
  clientWeight: number;
  dynamicData: boolean;
  dynamicClient?: number;
  receiverClient?: number;
  changeRound?: number;
}

interface GravityPanelProps {
  value: GravityHyperparams;
  onChange: (value: GravityHyperparams) => void;
  collapsed?: boolean;
  onCollapseToggle?: () => void;
}

export const GravityPanel: React.FC<GravityPanelProps> = ({ value, onChange, collapsed = false, onCollapseToggle }) => {
  const [local, setLocal] = useState(value);

  const handleChange = (field: keyof GravityHyperparams, val: any) => {
    const updated = { ...local, [field]: val };
    setLocal(updated);
    onChange(updated);
  };

  return (
    <Card className="mb-4 w-full border-primary/40">
      <CardHeader className="flex flex-row items-center justify-between cursor-pointer select-none" onClick={onCollapseToggle}>
        <CardTitle className="text-base flex items-center gap-2">
          Distances Inter-modèles – Hyperparamètres
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={onCollapseToggle}>
          {collapsed ? '▼' : '▲'}
        </Button>
      </CardHeader>
      {!collapsed && (
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3">
            <Switch checked={local.dynamicData} onCheckedChange={v => handleChange('dynamicData', v)} />
            <Label>Données dynamiques</Label>
          </div>
          {local.dynamicData && (
            <div className="flex gap-4">
              <div className="flex-1">
                <Label>Numéro du client dynamique</Label>
                <Input type="number" value={local.dynamicClient ?? ''} onChange={e => handleChange('dynamicClient', parseInt(e.target.value, 10))} />
              </div>
              <div className="flex-1">
                <Label>Numéro du paquet de données</Label>
                <Input type="number" value={local.receiverClient ?? ''} onChange={e => handleChange('receiverClient', parseInt(e.target.value, 10))} />
              </div>
              <div className="flex-1">
                <Label>Round de changement de données</Label>
                <Input type="number" value={local.changeRound ?? ''} onChange={e => handleChange('changeRound', parseInt(e.target.value, 10))} />
              </div>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
};

export default GravityPanel;
