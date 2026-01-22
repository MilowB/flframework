import React, { useState, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Play, Square, RotateCcw, Plus, Trash2, ChevronDown, Save, Zap, Users, FlaskConical } from 'lucide-react';
import { SidebarTrigger } from '@/components/ui/sidebar';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import { ServerConfig, FederatedState, ServerStatus, RoundMetrics, ClusterMetrics, ClientRoundMetrics } from '@/lib/federated/types';
import { aggregationMethods } from '@/lib/federated/aggregations';
import { preloadMNIST, initializeModel, runFederatedRound, createClient, setSeed } from '@/lib/federated/simulation';
import { saveExperiment, ExperimentData } from '@/lib/federated/experimentStorage';
import { SeededRandom } from '@/lib/federated/core/random';

// Strategy hyperparameters types
interface NoneHyperparams {
  dynamicData: boolean;
  dynamicClient?: number;
  receiverClient?: number;
  changeRound?: number;
}

interface FiftyFiftyHyperparams {
  dynamicData: boolean;
  dynamicClient?: number;
  receiverClient?: number;
  changeRound?: number;
}

interface GravityHyperparams {
  gravitationConstant: number;
  clusterWeight: number;
  clientWeight: number;
  dynamicData: boolean;
  dynamicClient?: number;
  receiverClient?: number;
  changeRound?: number;
}

interface StrategyHyperparams {
  none: NoneHyperparams;
  fiftyFifty: FiftyFiftyHyperparams;
  gravity: GravityHyperparams;
}

const defaultStrategyHyperparams: StrategyHyperparams = {
  none: { dynamicData: false },
  fiftyFifty: { dynamicData: false },
  gravity: {
    gravitationConstant: 1.0,
    clusterWeight: 1.0,
    clientWeight: 1.0,
    dynamicData: false,
  },
};

interface BenchmarkExperiment {
  id: string;
  name: string;
  config: ServerConfig;
  clientCount: number;
  isOpen: boolean;
  strategyHyperparams: StrategyHyperparams;
}

const architectures = [
  { value: 'mlp-small', label: 'MLP Small (784→128→10)' },
];

const clusteringMethods = [
  { value: 'louvain', label: 'Louvain' },
  { value: 'kmeans', label: 'K-means' },
  { value: 'leiden', label: 'Leiden' },
  { value: 'spectral', label: 'Spectral' },
];

const defaultConfig: ServerConfig = {
  aggregationMethod: 'fedavg',
  clientsPerRound: 6,
  totalRounds: 5,
  minClientsRequired: 2,
  modelArchitecture: 'mlp-small',
  seed: 42,
};

const generateId = () => Math.random().toString(36).substring(2, 9);

const Benchmark = () => {
  const [experiments, setExperiments] = useState<BenchmarkExperiment[]>([
    {
      id: generateId(),
      name: 'Expérience 1',
      config: { ...defaultConfig },
      clientCount: 6,
      isOpen: true,
      strategyHyperparams: { ...defaultStrategyHyperparams },
    },
  ]);

  const [masterSeed, setMasterSeed] = useState(42);
  const [seedCount, setSeedCountState] = useState(3);
  
  const [isRunning, setIsRunning] = useState(false);
  const [currentExperimentIndex, setCurrentExperimentIndex] = useState(0);
  const [currentSeedIndex, setCurrentSeedIndex] = useState(0);
  const [currentRound, setCurrentRound] = useState(0);
  const [totalRounds, setTotalRounds] = useState(0);
  const [completedResults, setCompletedResults] = useState<any[]>([]);
  const [mnistLoaded, setMnistLoaded] = useState(false);
  
  const abortRef = useRef(false);

  // Preload MNIST
  React.useEffect(() => {
    preloadMNIST().then(() => {
      setMnistLoaded(true);
    });
  }, []);

  const addExperiment = () => {
    setExperiments(prev => [
      ...prev,
      {
        id: generateId(),
        name: `Expérience ${prev.length + 1}`,
        config: { ...defaultConfig },
        clientCount: 6,
        isOpen: true,
        strategyHyperparams: { ...defaultStrategyHyperparams },
      },
    ]);
  };

  const removeExperiment = (id: string) => {
    if (experiments.length <= 1) {
      toast.error('Il faut au moins une expérience');
      return;
    }
    setExperiments(prev => prev.filter(e => e.id !== id));
  };

  const updateExperiment = (id: string, update: Partial<BenchmarkExperiment>) => {
    setExperiments(prev => prev.map(e => (e.id === id ? { ...e, ...update } : e)));
  };

  const updateExperimentConfig = (id: string, configUpdate: Partial<ServerConfig>) => {
    setExperiments(prev =>
      prev.map(e => (e.id === id ? { ...e, config: { ...e.config, ...configUpdate } } : e))
    );
  };

  const toggleExperiment = (id: string) => {
    setExperiments(prev => prev.map(e => (e.id === id ? { ...e, isOpen: !e.isOpen } : e)));
  };

  // Generate seeds from master seed
  const generateSeeds = useCallback((master: number, count: number): number[] => {
    const rng = new SeededRandom(master);
    return Array.from({ length: count }, () => rng.nextInt(100000));
  }, []);

  const startBenchmark = useCallback(async () => {
    if (!mnistLoaded) {
      toast.error('MNIST non chargé');
      return;
    }

    abortRef.current = false;
    setIsRunning(true);
    setCompletedResults([]);
    setCurrentExperimentIndex(0);
    setCurrentSeedIndex(0);
    setCurrentRound(0);

    const seeds = generateSeeds(masterSeed, seedCount);
    const totalExperiments = experiments.length * seeds.length;
    let expIdx = 0;

    // Group results by experiment for averaging
    const experimentResults: Map<string, { experiment: BenchmarkExperiment; seedResults: FederatedState[] }> = new Map();

    for (let ei = 0; ei < experiments.length; ei++) {
      if (abortRef.current) break;
      const experiment = experiments[ei];
      const seedStates: FederatedState[] = [];

      for (let si = 0; si < seeds.length; si++) {
        if (abortRef.current) break;
        const seed = seeds[si];
        
        setCurrentExperimentIndex(ei);
        setCurrentSeedIndex(si);
        setCurrentRound(0);
        setTotalRounds(experiment.config.totalRounds);

        // Initialize for this run
        setSeed(seed);
        const configWithSeed: ServerConfig = { ...experiment.config, seed };
        const globalModel = initializeModel(configWithSeed.modelArchitecture);
        const clients = Array.from({ length: experiment.clientCount }, (_, i) => createClient(i));

        let state: FederatedState = {
          isRunning: true,
          currentRound: 0,
          totalRounds: configWithSeed.totalRounds,
          clients,
          serverConfig: configWithSeed,
          roundHistory: [],
          globalModel,
          serverStatus: 'idle' as ServerStatus,
        };

        let clustersForRound: string[][] | undefined = undefined;

        // Run rounds
        for (let round = 0; round < configWithSeed.totalRounds; round++) {
          if (abortRef.current) break;
          setCurrentRound(round + 1);

          try {
            const [metrics, nextClusters] = await runFederatedRound(
              { ...state, currentRound: round },
              (update) => { state = { ...state, ...update }; },
              () => {},
              () => {},
              clustersForRound
            );
            clustersForRound = nextClusters;
            state.roundHistory = [...state.roundHistory, metrics];
          } catch (error) {
            console.error('Round failed:', error);
            break;
          }
        }

        seedStates.push(state);
      }

      experimentResults.set(experiment.id, { experiment, seedResults: seedStates });
    }

    // Convert to averaged results
    const averagedResults = Array.from(experimentResults.entries()).map(([expId, { experiment, seedResults }]) => {
      return {
        experimentId: expId,
        experimentName: experiment.name,
        config: experiment.config,
        clientCount: experiment.clientCount,
        seedCount: seedResults.length,
        seedResults,
      };
    });

    setCompletedResults(averagedResults);
    setIsRunning(false);
    
    if (!abortRef.current) {
      toast.success(`Benchmark terminé: ${averagedResults.length} expériences moyennées`);
    }
  }, [experiments, masterSeed, seedCount, mnistLoaded, generateSeeds]);

  const stopBenchmark = () => {
    abortRef.current = true;
    setIsRunning(false);
  };

  const resetBenchmark = () => {
    abortRef.current = true;
    setIsRunning(false);
    setCompletedResults([]);
    setCurrentExperimentIndex(0);
    setCurrentSeedIndex(0);
    setCurrentRound(0);
  };

  // Average round metrics across seeds
  const averageRoundHistory = (seedResults: FederatedState[]): RoundMetrics[] => {
    if (seedResults.length === 0) return [];
    
    const maxRounds = Math.max(...seedResults.map(s => s.roundHistory.length));
    const averaged: RoundMetrics[] = [];
    
    for (let r = 0; r < maxRounds; r++) {
      const roundData = seedResults
        .map(s => s.roundHistory[r])
        .filter(Boolean);
      
      if (roundData.length === 0) continue;
      
      // Average global metrics
      const avgLoss = roundData.reduce((sum, rd) => sum + rd.globalLoss, 0) / roundData.length;
      const avgAccuracy = roundData.reduce((sum, rd) => sum + rd.globalAccuracy, 0) / roundData.length;
      const avgSilhouette = roundData.reduce((sum, rd) => sum + (rd.silhouetteAvg || 0), 0) / roundData.length;
      
      // Average cluster metrics
      let avgClusterMetrics: ClusterMetrics[] | undefined;
      const allClusterMetrics = roundData.filter(rd => rd.clusterMetrics).map(rd => rd.clusterMetrics!);
      if (allClusterMetrics.length > 0) {
        const clusterCount = Math.max(...allClusterMetrics.map(cm => cm.length));
        avgClusterMetrics = [];
        for (let c = 0; c < clusterCount; c++) {
          const clusterData = allClusterMetrics.map(cm => cm[c]).filter(Boolean);
          if (clusterData.length > 0) {
            avgClusterMetrics.push({
              clusterId: c,
              accuracy: clusterData.reduce((sum, cd) => sum + cd.accuracy, 0) / clusterData.length,
              clientIds: clusterData[0].clientIds,
            });
          }
        }
      }
      
      // Average client metrics
      let avgClientMetrics: ClientRoundMetrics[] | undefined;
      const allClientMetrics = roundData.filter(rd => rd.clientMetrics).map(rd => rd.clientMetrics!);
      if (allClientMetrics.length > 0) {
        const clientIds = [...new Set(allClientMetrics.flatMap(cm => cm.map(c => c.clientId)))];
        avgClientMetrics = clientIds.map(clientId => {
          const clientData = allClientMetrics
            .flatMap(cm => cm.filter(c => c.clientId === clientId));
          return {
            clientId,
            clientName: clientData[0]?.clientName || clientId,
            loss: clientData.reduce((sum, cd) => sum + cd.loss, 0) / clientData.length,
            accuracy: clientData.reduce((sum, cd) => sum + cd.accuracy, 0) / clientData.length,
            testAccuracy: clientData.reduce((sum, cd) => sum + cd.testAccuracy, 0) / clientData.length,
          };
        });
      }
      
      averaged.push({
        round: r + 1,
        globalLoss: avgLoss,
        globalAccuracy: avgAccuracy,
        participatingClients: roundData[0].participatingClients,
        aggregationTime: roundData.reduce((sum, rd) => sum + rd.aggregationTime, 0) / roundData.length,
        timestamp: Date.now(),
        silhouetteAvg: avgSilhouette,
        clusterMetrics: avgClusterMetrics,
        clientMetrics: avgClientMetrics,
        distanceMatrix: roundData[0].distanceMatrix,
        clusters: roundData[0].clusters,
      });
    }
    
    return averaged;
  };

  const saveAllResults = () => {
    if (completedResults.length === 0) {
      toast.error('Aucun résultat à sauvegarder');
      return;
    }

    completedResults.forEach((result) => {
      const { experimentName, config, seedResults, seedCount: numSeeds } = result;
      
      // Average the round history across all seeds
      const averagedRoundHistory = averageRoundHistory(seedResults);
      
      // Use the global model from the last seed (or first available)
      const lastState = seedResults[seedResults.length - 1];
      
      const data: ExperimentData = {
        version: '1.0',
        savedAt: new Date().toISOString(),
        serverConfig: config,
        globalModel: lastState?.globalModel || null,
        roundHistory: averagedRoundHistory,
        clientModels: [],
      };

      const json = JSON.stringify(data, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const now = new Date();
      const timestamp = now.toISOString()
        .replace(/[:.]/g, '-')
        .replace('T', '_')
        .slice(0, 19);
      const filename = `benchmark-${experimentName.replace(/\s+/g, '-')}-avg${numSeeds}seeds-${timestamp}.json`;
      
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    toast.success(`${completedResults.length} fichiers moyennés téléchargés`);
  };

  const totalProgress = experiments.length * seedCount;
  const currentProgress = currentExperimentIndex * seedCount + currentSeedIndex + (currentRound > 0 ? 1 : 0);
  const progressPercent = totalProgress > 0 ? (currentProgress / totalProgress) * 100 : 0;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="container py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <SidebarTrigger className="h-8 w-8" />
              <div>
                <h1 className="text-xl font-bold text-foreground flex items-center gap-2">
                  <FlaskConical className="w-5 h-5 text-primary" />
                  Benchmark
                </h1>
                <p className="text-sm text-muted-foreground">
                  {mnistLoaded ? 'MNIST chargé' : 'Chargement MNIST...'} — {experiments.length} expérience(s), {seedCount} seed(s)
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container py-6 space-y-6">
        {/* Control Panel */}
        <div className="p-6 rounded-xl bg-gradient-card border border-border shadow-card">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            {/* Left: Controls */}
            <div className="flex items-center gap-3">
              <Button
                onClick={isRunning ? stopBenchmark : startBenchmark}
                disabled={!mnistLoaded}
                className={cn(
                  'gap-2 min-w-[140px] font-medium transition-all',
                  isRunning 
                    ? 'bg-destructive hover:bg-destructive/90' 
                    : 'bg-gradient-primary hover:opacity-90'
                )}
              >
                {isRunning ? (
                  <>
                    <Square className="w-4 h-4" />
                    Arrêter
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Démarrer
                  </>
                )}
              </Button>
              
              <Button
                variant="outline"
                onClick={resetBenchmark}
                disabled={isRunning}
                className="gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                Réinitialiser
              </Button>

              <Button
                variant="outline"
                onClick={saveAllResults}
                disabled={isRunning || completedResults.length === 0}
                className="gap-2"
              >
                <Save className="w-4 h-4" />
                Sauvegarder ({completedResults.length})
              </Button>
            </div>

            {/* Center: Progress */}
            <div className="flex-1 max-w-md">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Progression globale</span>
                <span className="font-mono text-sm">
                  Exp <span className="text-primary">{currentExperimentIndex + 1}</span>/{experiments.length} 
                  {' '}— Seed <span className="text-primary">{currentSeedIndex + 1}</span>/{seedCount}
                  {' '}— Round <span className="text-primary">{currentRound}</span>/{totalRounds}
                </span>
              </div>
              <Progress value={progressPercent} className="h-2" />
            </div>

            {/* Right: Seeds */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground whitespace-nowrap">Seeds</Label>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={seedCount}
                  onChange={e => setSeedCountState(Math.max(1, Math.min(20, Number(e.target.value))))}
                  disabled={isRunning}
                  className="w-16"
                />
              </div>

              <div className="flex items-center gap-2">
                <Label className="text-sm text-muted-foreground whitespace-nowrap">Master Seed</Label>
                <Input
                  type="number"
                  min={0}
                  value={masterSeed}
                  onChange={e => setMasterSeed(Number(e.target.value))}
                  disabled={isRunning}
                  className="w-24"
                />
              </div>
            </div>

            {/* Status Indicator */}
            {isRunning && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/30">
                <Zap className="w-4 h-4 text-primary animate-pulse" />
                <span className="text-sm font-medium text-primary">En cours</span>
              </div>
            )}
          </div>
        </div>

        {/* Experiment Panels */}
        <div className="space-y-4">
          {experiments.map((experiment, index) => (
            <ExperimentPanel
              key={experiment.id}
              experiment={experiment}
              index={index}
              disabled={isRunning}
              onToggle={() => toggleExperiment(experiment.id)}
              onRemove={() => removeExperiment(experiment.id)}
              onUpdateName={(name) => updateExperiment(experiment.id, { name })}
              onUpdateConfig={(config) => updateExperimentConfig(experiment.id, config)}
              onUpdateClientCount={(count) => updateExperiment(experiment.id, { clientCount: count })}
              onUpdateStrategyHyperparams={(strategyHyperparams) => updateExperiment(experiment.id, { strategyHyperparams })}
            />
          ))}

          {/* Add Experiment Button */}
          <Button
            variant="outline"
            onClick={addExperiment}
            disabled={isRunning}
            className="w-full gap-2 border-dashed h-14"
          >
            <Plus className="w-5 h-5" />
            Ajouter une expérience
          </Button>
        </div>
      </main>
    </div>
  );
};

interface ExperimentPanelProps {
  experiment: BenchmarkExperiment;
  index: number;
  disabled: boolean;
  onToggle: () => void;
  onRemove: () => void;
  onUpdateName: (name: string) => void;
  onUpdateConfig: (config: Partial<ServerConfig>) => void;
  onUpdateClientCount: (count: number) => void;
  onUpdateStrategyHyperparams: (hyperparams: StrategyHyperparams) => void;
}

const ExperimentPanel = ({
  experiment,
  index,
  disabled,
  onToggle,
  onRemove,
  onUpdateName,
  onUpdateConfig,
  onUpdateClientCount,
  onUpdateStrategyHyperparams,
}: ExperimentPanelProps) => {
  const { config, strategyHyperparams } = experiment;
  const clientAggMethod = config.clientAggregationMethod ?? 'none';

  const updateNoneParams = (update: Partial<NoneHyperparams>) => {
    onUpdateStrategyHyperparams({
      ...strategyHyperparams,
      none: { ...strategyHyperparams.none, ...update },
    });
  };

  const updateFiftyFiftyParams = (update: Partial<FiftyFiftyHyperparams>) => {
    onUpdateStrategyHyperparams({
      ...strategyHyperparams,
      fiftyFifty: { ...strategyHyperparams.fiftyFifty, ...update },
    });
  };

  const updateGravityParams = (update: Partial<GravityHyperparams>) => {
    onUpdateStrategyHyperparams({
      ...strategyHyperparams,
      gravity: { ...strategyHyperparams.gravity, ...update },
    });
  };

  return (
    <Collapsible open={experiment.isOpen} onOpenChange={onToggle}>
      <Card className="bg-gradient-card border-border shadow-card">
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                  <span className="text-sm font-bold text-primary">{index + 1}</span>
                </div>
                <Input
                  value={experiment.name}
                  onChange={e => onUpdateName(e.target.value)}
                  onClick={e => e.stopPropagation()}
                  disabled={disabled}
                  className="w-48 bg-transparent border-transparent hover:border-border focus:border-primary"
                />
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove();
                  }}
                  disabled={disabled}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                <ChevronDown className={cn(
                  'w-5 h-5 text-muted-foreground transition-transform',
                  experiment.isOpen && 'rotate-180'
                )} />
              </div>
            </CardTitle>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="pt-0">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Model Architecture */}
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Architecture</Label>
                <Select
                  value={config.modelArchitecture}
                  onValueChange={(value) => onUpdateConfig({ modelArchitecture: value })}
                  disabled={disabled}
                >
                  <SelectTrigger className="bg-muted/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {architectures.map((arch) => (
                      <SelectItem key={arch.value} value={arch.value}>
                        {arch.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Server Aggregation */}
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Agrégation serveur</Label>
                <Select
                  value={config.aggregationMethod}
                  onValueChange={(value: ServerConfig['aggregationMethod']) =>
                    onUpdateConfig({ aggregationMethod: value })
                  }
                  disabled={disabled}
                >
                  <SelectTrigger className="bg-muted/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(aggregationMethods).map(([key, method]) => (
                      <SelectItem key={key} value={key}>
                        {method.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Client Aggregation */}
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Agrégation client</Label>
                <Select
                  value={config.clientAggregationMethod ?? 'none'}
                  onValueChange={(value: 'none' | '50-50' | 'gravity') =>
                    onUpdateConfig({ clientAggregationMethod: value })
                  }
                  disabled={disabled}
                >
                  <SelectTrigger className="bg-muted/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="50-50">50/50</SelectItem>
                    <SelectItem value="gravity">Distances Inter-modèles</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Model Assignment */}
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Affectation modèle</Label>
                <Select
                  value={config.modelAssignmentMethod ?? '1NN'}
                  onValueChange={(value: '1NN' | 'Probabiliste') =>
                    onUpdateConfig({ modelAssignmentMethod: value })
                  }
                  disabled={disabled}
                >
                  <SelectTrigger className="bg-muted/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1NN">1NN</SelectItem>
                    <SelectItem value="Probabiliste">Probabiliste</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Clustering Method */}
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Méthode de clustering</Label>
                <Select
                  value={config.clusteringMethod ?? 'louvain'}
                  onValueChange={(value: 'louvain' | 'kmeans' | 'leiden' | 'spectral') =>
                    onUpdateConfig({ clusteringMethod: value })
                  }
                  disabled={disabled}
                >
                  <SelectTrigger className="bg-muted/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {clusteringMethods.map((method) => (
                      <SelectItem key={method.value} value={method.value}>
                        {method.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* K-means / Spectral Number of Clusters */}
              {(config.clusteringMethod === 'kmeans' || config.clusteringMethod === 'spectral') && (
                <div className="space-y-2">
                  <Label className="flex items-center justify-between text-sm text-muted-foreground">
                    <span>Nb clusters (K)</span>
                    <span className="font-mono text-primary">{config.clusteringMethod === 'kmeans' ? (config.kmeansNumClusters ?? 3) : (config.spectralNumClusters ?? 3)}</span>
                  </Label>
                  <Slider
                    value={[config.clusteringMethod === 'kmeans' ? (config.kmeansNumClusters ?? 3) : (config.spectralNumClusters ?? 3)]}
                    onValueChange={([value]) => onUpdateConfig(
                      config.clusteringMethod === 'kmeans' 
                        ? { kmeansNumClusters: value } 
                        : { spectralNumClusters: value }
                    )}
                    min={2}
                    max={10}
                    step={1}
                    disabled={disabled}
                    className="[&_[role=slider]]:bg-primary"
                  />
                </div>
              )}

              {/* Total Rounds */}
              <div className="space-y-2">
                <Label className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Rounds</span>
                  <span className="font-mono text-primary">{config.totalRounds}</span>
                </Label>
                <Slider
                  value={[config.totalRounds]}
                  onValueChange={([value]) => onUpdateConfig({ totalRounds: value })}
                  min={1}
                  max={50}
                  step={1}
                  disabled={disabled}
                  className="[&_[role=slider]]:bg-primary"
                />
              </div>

              {/* Clients per Round */}
              <div className="space-y-2">
                <Label className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Clients/round</span>
                  <span className="font-mono text-primary">{config.clientsPerRound}</span>
                </Label>
                <Slider
                  value={[config.clientsPerRound]}
                  onValueChange={([value]) => onUpdateConfig({ clientsPerRound: value })}
                  min={1}
                  max={10}
                  step={1}
                  disabled={disabled}
                  className="[&_[role=slider]]:bg-primary"
                />
              </div>

              {/* Min Clients */}
              <div className="space-y-2">
                <Label className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Clients min</span>
                  <span className="font-mono text-primary">{config.minClientsRequired}</span>
                </Label>
                <Slider
                  value={[config.minClientsRequired]}
                  onValueChange={([value]) => onUpdateConfig({ minClientsRequired: value })}
                  min={1}
                  max={config.clientsPerRound}
                  step={1}
                  disabled={disabled}
                  className="[&_[role=slider]]:bg-primary"
                />
              </div>

              {/* Client Count (per experiment) */}
              <div className="space-y-2">
                <Label className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Nb clients</span>
                  <span className="font-mono text-primary">{experiment.clientCount}</span>
                </Label>
                <Slider
                  value={[experiment.clientCount]}
                  onValueChange={([value]) => onUpdateClientCount(value)}
                  min={2}
                  max={12}
                  step={1}
                  disabled={disabled}
                  className="[&_[role=slider]]:bg-primary"
                />
              </div>
            </div>

            {/* Strategy Hyperparameters Panel */}
            {clientAggMethod === 'none' && (
              <div className="mt-6 p-4 rounded-lg border border-primary/30 bg-primary/5">
                <h4 className="text-sm font-medium mb-4 text-primary">None – Hyperparamètres</h4>
                <div className="flex items-center gap-3 mb-4">
                  <Switch 
                    checked={strategyHyperparams.none.dynamicData} 
                    onCheckedChange={v => updateNoneParams({ dynamicData: v })} 
                    disabled={disabled}
                  />
                  <Label className="text-sm">Données dynamiques</Label>
                </div>
                {strategyHyperparams.none.dynamicData && (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label className="text-xs text-muted-foreground">Client dynamique</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.none.dynamicClient ?? ''} 
                        onChange={e => updateNoneParams({ dynamicClient: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Paquet de données</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.none.receiverClient ?? ''} 
                        onChange={e => updateNoneParams({ receiverClient: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Round de changement</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.none.changeRound ?? ''} 
                        onChange={e => updateNoneParams({ changeRound: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {clientAggMethod === '50-50' && (
              <div className="mt-6 p-4 rounded-lg border border-primary/30 bg-primary/5">
                <h4 className="text-sm font-medium mb-4 text-primary">50/50 – Hyperparamètres</h4>
                <div className="flex items-center gap-3 mb-4">
                  <Switch 
                    checked={strategyHyperparams.fiftyFifty.dynamicData} 
                    onCheckedChange={v => updateFiftyFiftyParams({ dynamicData: v })} 
                    disabled={disabled}
                  />
                  <Label className="text-sm">Données dynamiques</Label>
                </div>
                {strategyHyperparams.fiftyFifty.dynamicData && (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label className="text-xs text-muted-foreground">Client dynamique</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.fiftyFifty.dynamicClient ?? ''} 
                        onChange={e => updateFiftyFiftyParams({ dynamicClient: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Paquet de données</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.fiftyFifty.receiverClient ?? ''} 
                        onChange={e => updateFiftyFiftyParams({ receiverClient: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Round de changement</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.fiftyFifty.changeRound ?? ''} 
                        onChange={e => updateFiftyFiftyParams({ changeRound: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {clientAggMethod === 'gravity' && (
              <div className="mt-6 p-4 rounded-lg border border-primary/30 bg-primary/5">
                <h4 className="text-sm font-medium mb-4 text-primary">Distances Inter-modèles – Hyperparamètres</h4>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Constante gravitation (G)</Label>
                    <Input 
                      type="number" 
                      step="any"
                      value={strategyHyperparams.gravity.gravitationConstant} 
                      onChange={e => updateGravityParams({ gravitationConstant: parseFloat(e.target.value) })} 
                      disabled={disabled}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Poids du cluster</Label>
                    <Input 
                      type="number" 
                      step="any"
                      value={strategyHyperparams.gravity.clusterWeight} 
                      onChange={e => updateGravityParams({ clusterWeight: parseFloat(e.target.value) })} 
                      disabled={disabled}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Poids du client</Label>
                    <Input 
                      type="number" 
                      step="any"
                      value={strategyHyperparams.gravity.clientWeight} 
                      onChange={e => updateGravityParams({ clientWeight: parseFloat(e.target.value) })} 
                      disabled={disabled}
                      className="mt-1"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-3 mb-4">
                  <Switch 
                    checked={strategyHyperparams.gravity.dynamicData} 
                    onCheckedChange={v => updateGravityParams({ dynamicData: v })} 
                    disabled={disabled}
                  />
                  <Label className="text-sm">Données dynamiques</Label>
                </div>
                {strategyHyperparams.gravity.dynamicData && (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label className="text-xs text-muted-foreground">Client dynamique</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.gravity.dynamicClient ?? ''} 
                        onChange={e => updateGravityParams({ dynamicClient: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Paquet de données</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.gravity.receiverClient ?? ''} 
                        onChange={e => updateGravityParams({ receiverClient: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Round de changement</Label>
                      <Input 
                        type="number" 
                        value={strategyHyperparams.gravity.changeRound ?? ''} 
                        onChange={e => updateGravityParams({ changeRound: e.target.value === '' ? undefined : parseInt(e.target.value, 10) })} 
                        disabled={disabled}
                        className="mt-1"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
};

export default Benchmark;
