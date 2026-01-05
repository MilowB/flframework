import { useFederatedLearning } from '@/hooks/useFederatedLearning';
import { ServerPanel } from '@/components/federated/ServerPanel';
import { ClientCard } from '@/components/federated/ClientCard';
import { MetricsChart } from '@/components/federated/MetricsChart';
import { WeightsChart } from '@/components/federated/WeightsChart';
import { ControlPanel } from '@/components/federated/ControlPanel';
import { NetworkVisualization } from '@/components/federated/NetworkVisualization';
import { RoundHistory } from '@/components/federated/RoundHistory';
import { CodePreview } from '@/components/federated/CodePreview';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Network, LayoutGrid, Code } from 'lucide-react';

const Index = () => {
  const {
    state,
    startTraining,
    stopTraining,
    resetTraining,
    setClientCount,
    updateServerConfig,
  } = useFederatedLearning(6);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="container py-4">
          <div className="flex items-center gap-4">
            <div className="p-2.5 rounded-xl bg-gradient-primary">
              <Network className="w-6 h-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">
                Federated Learning Framework
              </h1>
              <p className="text-sm text-muted-foreground">
                Simulation d'apprentissage fédéré distribué
              </p>
            </div>
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
          <div className="lg:col-span-6 space-y-6">
            <Tabs defaultValue="network" className="w-full">
              <TabsList className="w-full grid grid-cols-3 bg-muted/30">
                <TabsTrigger value="network" className="gap-2">
                  <Network className="w-4 h-4" />
                  Réseau
                </TabsTrigger>
                <TabsTrigger value="clients" className="gap-2">
                  <LayoutGrid className="w-4 h-4" />
                  Clients
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

              <TabsContent value="clients" className="mt-4">
                <ScrollArea className="h-[400px] pr-4">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
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

            <MetricsChart history={state.roundHistory} />
            <WeightsChart history={state.roundHistory} />
          </div>

          {/* Right Sidebar - History */}
          <div className="lg:col-span-3">
            <RoundHistory history={state.roundHistory} />
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

export default Index;
