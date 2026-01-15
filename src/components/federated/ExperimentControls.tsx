import { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Save, Upload } from 'lucide-react';
import { FederatedState, ModelWeights } from '@/lib/federated/types';
import { saveExperiment, loadExperiment, ExperimentData } from '@/lib/federated/experimentStorage';
import { Model3DPosition } from '@/lib/federated/visualization/pca';
import { toast } from 'sonner';

interface ExperimentControlsProps {
  state: FederatedState;
  clientModels: Map<string, ModelWeights>;
  onLoad: (data: ExperimentData) => void;
  disabled?: boolean;
  visualizations3D?: { round: number; models: Model3DPosition[] }[];
}

export const ExperimentControls = ({ 
  state, 
  clientModels, 
  onLoad,
  disabled,
  visualizations3D
}: ExperimentControlsProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSave = () => {
    if (state.roundHistory.length === 0) {
      toast.error('Aucune donnée à sauvegarder');
      return;
    }
    saveExperiment(state, clientModels, visualizations3D);
    toast.success('Expérience sauvegardée');
  };

  const handleLoadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const data = await loadExperiment(file);
      onLoad(data);
      toast.success(`Expérience chargée: ${data.roundHistory.length} rounds`);
    } catch (err) {
      toast.error((err as Error).message);
    }

    // Reset input
    e.target.value = '';
  };

  const hasData = state.roundHistory.length > 0;

  return (
    <div className="flex items-center gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={handleSave}
        disabled={disabled || !hasData}
        className="gap-2"
      >
        <Save className="w-4 h-4" />
        Sauvegarder
      </Button>
      
      <Button
        variant="outline"
        size="sm"
        onClick={handleLoadClick}
        disabled={disabled}
        className="gap-2"
      >
        <Upload className="w-4 h-4" />
        Charger
      </Button>

      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
};
