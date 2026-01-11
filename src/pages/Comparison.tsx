import { useState, useRef, useMemo } from 'react';
import { Upload, X, FileJson, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { loadExperiment, type ExperimentData } from '@/lib/federated/results/experimentStorage';
import { useToast } from '@/hooks/use-toast';
import { ComparisonChart } from '@/components/federated/ComparisonChart';
import { ComparisonSimilarityMatrix } from '@/components/federated/ComparisonSimilarityMatrix';

interface LoadedExperiment {
  id: string;
  name: string;
  data: ExperimentData;
}

const Comparison = () => {
  const [experiments, setExperiments] = useState<LoadedExperiment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileLoad = async (file: File) => {
    try {
      const data = await loadExperiment(file);
      const newExperiment: LoadedExperiment = {
        id: crypto.randomUUID(),
        name: file.name.replace('.json', ''),
        data,
      };
      setExperiments((prev) => [...prev, newExperiment]);
      toast({
        title: 'Expérience chargée',
        description: `${file.name} ajoutée à la comparaison.`,
      });
    } catch (error) {
      toast({
        title: 'Erreur',
        description: 'Impossible de charger le fichier d\'expérience.',
        variant: 'destructive',
      });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      Array.from(files).forEach(handleFileLoad);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeExperiment = (id: string) => {
    setExperiments((prev) => prev.filter((exp) => exp.id !== id));
  };

  const clearAllExperiments = () => {
    setExperiments([]);
    toast({
      title: 'Comparaison vidée',
      description: 'Toutes les expériences ont été retirées.',
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-12 z-30 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="container py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-foreground">
                Comparaison d'expériences
              </h1>
              <p className="text-sm text-muted-foreground">
                Chargez des fichiers d'expériences pour comparer les résultats
              </p>
            </div>
            <div className="flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                multiple
                onChange={handleFileChange}
                className="hidden"
              />
              {experiments.length > 0 && (
                <Button
                  variant="outline"
                  onClick={clearAllExperiments}
                  className="gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Vider
                </Button>
              )}
              <Button
                onClick={() => fileInputRef.current?.click()}
                className="gap-2"
              >
                <Upload className="w-4 h-4" />
                Charger une expérience
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container py-6 space-y-6">
        {/* Loaded experiments cards */}
        {experiments.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {experiments.map((exp, idx) => (
              <Card key={exp.id} className="bg-gradient-card border-border">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileJson className="w-4 h-4 text-primary" />
                      <CardTitle className="text-sm font-medium truncate">
                        Exp {idx + 1}: {exp.name}
                      </CardTitle>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={() => removeExperiment(exp.id)}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-muted-foreground space-y-1">
                    <p>Rounds: {exp.data.roundHistory.length}</p>
                    <p>Clients: {exp.data.clientModels.length}</p>
                    <p>Agrégation: {exp.data.serverConfig.aggregationMethod}</p>
                    <p>Sauvegardé: {new Date(exp.data.savedAt).toLocaleString()}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Empty state */}
        {experiments.length === 0 && (
          <Card className="bg-gradient-card border-border border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-16">
              <Upload className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">
                Aucune expérience chargée
              </h3>
              <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
                Chargez des fichiers JSON d'expériences pour comparer leurs résultats côte à côte.
              </p>
              <Button onClick={() => fileInputRef.current?.click()} className="gap-2">
                <Upload className="w-4 h-4" />
                Charger des expériences
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Comparison charts */}
        {experiments.length > 0 && (
          <>
            <ComparisonChart experiments={experiments.map((e) => ({ name: e.name, data: e.data }))} />
            <ComparisonSimilarityMatrix experiments={experiments.map((e) => ({ name: e.name, data: e.data }))} />
          </>
        )}
      </main>
    </div>
  );
};

export default Comparison;