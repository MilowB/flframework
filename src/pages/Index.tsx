import React from 'react';
import { useFederatedLearning } from '@/hooks/useFederatedLearning';
import { ServerPanel } from '@/components/federated/ServerPanel';
import { ClientCard } from '@/components/federated/ClientCard';
import { ServerCard } from '@/components/federated/ServerCard';
import { MetricsChart } from '@/components/federated/MetricsChart';
import { ControlPanel } from '@/components/federated/ControlPanel';
import { NetworkVisualization } from '@/components/federated/NetworkVisualization';
import { RoundHistory } from '@/components/federated/RoundHistory';
import { CodePreview } from '@/components/federated/CodePreview';
import { ExperimentControls } from '@/components/federated/ExperimentControls';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Network, LayoutGrid, Code, Menu } from 'lucide-react';
import { SidebarTrigger } from '@/components/ui/sidebar';
import { StrategyHyperparamsProvider } from '@/components/federated/StrategyHyperparamsProvider';
import { useStrategyHyperparams } from '@/components/federated/StrategyHyperparamsContext';
import GravityPanel from '@/components/federated/GravityPanel';
import NonePanel from '@/components/federated/NonePanel';
import FiftyFiftyPanel from '@/components/federated/FiftyFiftyPanel';
import KmeansPanel from '@/components/federated/KmeansPanel';


const IndexContent = () => {
  const { gravity, setGravity } = useStrategyHyperparams();
  // Ajout des états locaux pour les autres stratégies dynamiques
  const [none, setNone] = React.useState({ dynamicData: false });
  const [fiftyFifty, setFiftyFifty] = React.useState({ dynamicData: false });
  const [kmeans, setKmeans] = React.useState({ numClusters: 3 });
  const {
    state,
    mnistLoaded,
    clientModels,
    startTraining,
    stopTraining,
    resetTraining,
    setClientCount,
    updateServerConfig,
    loadExperiment,
  } = useFederatedLearning(6, gravity, none, fiftyFifty);
  const [gravityCollapsed, setGravityCollapsed] = React.useState(false);
  const [kmeansCollapsed, setKmeansCollapsed] = React.useState(false);

  // Determine if gravity strategy is selected
  const isNone = state.serverConfig?.clientAggregationMethod === 'none';
  const isFiftyFifty = state.serverConfig?.clientAggregationMethod === '50-50';
  const isGravity = state.serverConfig?.clientAggregationMethod === 'gravity';
  const isKmeans = state.serverConfig?.clusteringMethod === 'kmeans';

  // Sync kmeans numClusters with serverConfig
  React.useEffect(() => {
    if (isKmeans && kmeans.numClusters !== state.serverConfig?.kmeansNumClusters) {
      updateServerConfig({ kmeansNumClusters: kmeans.numClusters });
    }
  }, [kmeans.numClusters, isKmeans]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="container py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <SidebarTrigger className="h-8 w-8" />
              <div>
                <h1 className="text-xl font-bold text-foreground">
                  Federated Learning Framework
                </h1>
                <p className="text-sm text-muted-foreground">
                  MNIST Classification — {mnistLoaded ? '60,000 images chargées' : 'Chargement MNIST...'}
                </p>
              </div>
            </div>
            <ExperimentControls
              state={state}
              clientModels={clientModels}
              onLoad={loadExperiment}
              disabled={state.isRunning}
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-6 space-y-6">
        {/* Control Panel */}
        <ControlPanel
          isRunning={state.isRunning}
          currentRound={state.currentRound}
          totalRounds={state.totalRounds}
          clientCount={state.clients.length}
          onStart={startTraining}
          onStop={stopTraining}
          onReset={resetTraining}
          onClientCountChange={setClientCount}
        />

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar - Server Config */}
          <div className="lg:col-span-3">
            <ServerPanel
              config={state.serverConfig}
              onConfigChange={updateServerConfig}
              disabled={state.isRunning}
              globalModelVersion={state.globalModel?.version ?? 0}
            />
          </div>

          {/* Center - Visualization & Metrics */}
          <div className="lg:col-span-9 space-y-6">
            {/* Client Aggregation Panels */}
            {isNone && (
              <NonePanel
                value={none}
                onChange={setNone}
                collapsed={gravityCollapsed}
                onCollapseToggle={() => setGravityCollapsed((c) => !c)}
              />
            )}
            {isFiftyFifty && (
              <FiftyFiftyPanel
                value={fiftyFifty}
                onChange={setFiftyFifty}
                collapsed={gravityCollapsed}
                onCollapseToggle={() => setGravityCollapsed((c) => !c)}
              />
            )}
            {isGravity && (
              <GravityPanel
                value={gravity}
                onChange={setGravity}
                collapsed={gravityCollapsed}
                onCollapseToggle={() => setGravityCollapsed((c) => !c)}
              />
            )}
            {isKmeans && (
              <KmeansPanel
                value={kmeans}
                onChange={setKmeans}
                collapsed={kmeansCollapsed}
                onCollapseToggle={() => setKmeansCollapsed((c) => !c)}
              />
            )}
            <Tabs defaultValue="network" className="w-full">
              <TabsList className="w-full grid grid-cols-3 bg-muted/30">
                <TabsTrigger value="network" className="gap-2">
                  <Network className="w-4 h-4" />
                  Réseau
                </TabsTrigger>
                <TabsTrigger value="machines" className="gap-2">
                  <LayoutGrid className="w-4 h-4" />
                  Machines
                </TabsTrigger>
                <TabsTrigger value="code" className="gap-2">
                  <Code className="w-4 h-4" />
                  API
                </TabsTrigger>
              </TabsList>

              <TabsContent value="network" className="mt-4">
                <NetworkVisualization
                  clients={state.clients}
                  globalModelVersion={state.globalModel?.version ?? 0}
                />
              </TabsContent>

              <TabsContent value="machines" className="mt-4">
                <ScrollArea className="h-[400px] pr-4">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <ServerCard 
                      status={state.serverStatus} 
                      modelVersion={state.globalModel?.version ?? 0}
                    />
                    {state.clients.map((client) => (
                      <ClientCard key={client.id} client={client} />
                    ))}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="code" className="mt-4">
                <CodePreview />
              </TabsContent>
            </Tabs>

            <MetricsChart 
              history={state.roundHistory}
              clientModels={new Map(
                state.clients.map(c => [
                  c.id, 
                  { 
                    weights: c.lastLocalModel || { layers: [], bias: [] }, 
                    name: c.name 
                  }
                ])
              )}
              clusterModels={clientModels}
              globalModel={state.globalModel}
            />
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/20 py-6 mt-12">
        <div className="container">
          <p className="text-center text-sm text-muted-foreground">
            Framework d'Apprentissage Fédéré — Configurable, Extensible, Open Source
          </p>
        </div>
      </footer>
    </div>
  );
};

const Index = () => (
  <StrategyHyperparamsProvider>
    <IndexContent />
  </StrategyHyperparamsProvider>
);

export default Index;
