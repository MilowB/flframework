import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface KmeansHyperparams {
  numClusters: number;
}

interface KmeansPanelProps {
  value: KmeansHyperparams;
  onChange: (value: KmeansHyperparams) => void;
  collapsed?: boolean;
  onCollapseToggle?: () => void;
}

export const KmeansPanel: React.FC<KmeansPanelProps> = ({ value, onChange, collapsed = false, onCollapseToggle }) => {
  const [local, setLocal] = useState(value);

  const handleChange = (field: keyof KmeansHyperparams, val: any) => {
    const updated = { ...local, [field]: val };
    setLocal(updated);
    onChange(updated);
  };

  return (
    <Card className="mb-4 mx-auto max-w-xl border-primary/40">
      <CardHeader className="flex flex-row items-center justify-between cursor-pointer select-none" onClick={onCollapseToggle}>
        <CardTitle className="text-base flex items-center gap-2">
          K-means Clustering – Hyperparamètres
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={onCollapseToggle}>
          {collapsed ? '▼' : '▲'}
        </Button>
      </CardHeader>
      {!collapsed && (
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <Label>Nombre de clusters (K)</Label>
              <Input 
                type="number" 
                min={1}
                max={10}
                value={local.numClusters ?? 3} 
                onChange={e => handleChange('numClusters', parseInt(e.target.value, 10) || 3)} 
              />
              <p className="text-xs text-muted-foreground mt-1">
                Laissez vide pour détection automatique
              </p>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
};

export default KmeansPanel;
