import { ServerConfig } from '@/lib/federated/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Settings, Ruler, GitMerge } from 'lucide-react';

interface ClientSettingsPanelProps {
  config: ServerConfig;
  onConfigChange: (config: Partial<ServerConfig>) => void;
  disabled?: boolean;
}

export const ClientSettingsPanel = ({ config, onConfigChange, disabled }: ClientSettingsPanelProps) => {
  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="p-2 rounded-lg bg-primary/10">
            <Settings className="w-5 h-5 text-primary" />
          </div>
          <span>Paramètres Client</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Métrique de distance */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm text-muted-foreground">
            <Ruler className="w-4 h-4" />
            Métrique de distance
          </Label>
          <Select
            value={config.distanceMetric ?? 'cosine'}
            onValueChange={(value: 'l1' | 'l2' | 'cosine') => onConfigChange({ distanceMetric: value })}
            disabled={disabled}
          >
            <SelectTrigger className="bg-muted/50 border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="l1">
                <span className="font-medium">L1 (Manhattan)</span>
              </SelectItem>
              <SelectItem value="l2">
                <span className="font-medium">L2 (Euclidienne)</span>
              </SelectItem>
              <SelectItem value="cosine">
                <span className="font-medium">Cosine Similarity</span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Méthode d'agrégation client */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm text-muted-foreground">
            <GitMerge className="w-4 h-4" />
            Méthode d'agrégation client
          </Label>
          <Select
            value={config.clientAggregationMethod ?? 'none'}
            onValueChange={(value: 'none' | '50-50' | 'gravity') => onConfigChange({ clientAggregationMethod: value })}
            disabled={disabled}
          >
            <SelectTrigger className="bg-muted/50 border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">
                <span className="font-medium">None</span>
              </SelectItem>
              <SelectItem value="50-50">
                <span className="font-medium">50/50</span>
              </SelectItem>
              <SelectItem value="gravity">
                <span className="font-medium">Norme du gradient</span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
};