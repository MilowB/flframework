import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Code, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

export const CodePreview = () => {
  const [copiedTab, setCopiedTab] = useState<string | null>(null);

  const copyCode = (code: string, tab: string) => {
    navigator.clipboard.writeText(code);
    setCopiedTab(tab);
    setTimeout(() => setCopiedTab(null), 2000);
  };

  const clientCode = `// Définition du comportement client personnalisé
const customClientBehavior: ClientBehavior = {
  onReceiveModel: async (model: ModelWeights) => {
    console.log('Modèle reçu, version:', model.version);
    // Charger le modèle dans le framework ML local
    await loadModel(model);
  },
  
  onTrain: async (model, epochs) => {
    // Entraînement local sur les données du client
    const result = await trainLocal(model, {
      epochs,
      learningRate: 0.01,
      batchSize: 32,
    });
    
    return {
      weights: result.newWeights,
      loss: result.finalLoss,
      accuracy: result.accuracy,
    };
  },
  
  onSendModel: async (weights) => {
    // Envoyer les poids mis à jour au serveur
    await sendToServer(weights);
  },
};`;

  const aggregationCode = `// Fonction d'agrégation personnalisée
const customAggregation: AggregationFunction = (clientWeights) => {
  // Exemple: Agrégation avec pondération par qualité
  const qualityScores = clientWeights.map(c => 
    1 / (c.loss + 0.01) // Plus la loss est basse, plus le poids est élevé
  );
  const totalScore = qualityScores.reduce((a, b) => a + b, 0);
  
  const aggregated = initializeEmptyWeights();
  
  clientWeights.forEach((client, idx) => {
    const weight = qualityScores[idx] / totalScore;
    addWeightedModel(aggregated, client.weights, weight);
  });
  
  return aggregated;
};

// Enregistrer la fonction personnalisée
registerAggregation('quality-weighted', customAggregation);`;

  const serverCode = `// Configuration du serveur
const serverConfig: ServerConfig = {
  aggregationMethod: 'fedavg',  // ou 'custom'
  clientsPerRound: 5,
  totalRounds: 100,
  minClientsRequired: 3,
  modelArchitecture: 'resnet-mini',
};

// Lancer le serveur fédéré
const server = new FederatedServer(serverConfig);

// Ajouter des clients
clients.forEach(client => server.registerClient(client));

// Démarrer l'entraînement
await server.startTraining({
  onRoundComplete: (metrics) => {
    console.log(\`Round \${metrics.round}: Loss=\${metrics.loss}\`);
  },
  onTrainingComplete: (finalModel) => {
    saveModel(finalModel, 'trained_model.pt');
  },
});`;

  return (
    <Card className="bg-gradient-card border-border shadow-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Code className="w-5 h-5 text-primary" />
          API du Framework
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="client" className="w-full">
          <TabsList className="w-full grid grid-cols-3 bg-muted/30">
            <TabsTrigger value="client" className="text-xs">Client</TabsTrigger>
            <TabsTrigger value="aggregation" className="text-xs">Agrégation</TabsTrigger>
            <TabsTrigger value="server" className="text-xs">Serveur</TabsTrigger>
          </TabsList>
          
          {[
            { value: 'client', code: clientCode },
            { value: 'aggregation', code: aggregationCode },
            { value: 'server', code: serverCode },
          ].map(({ value, code }) => (
            <TabsContent key={value} value={value} className="mt-3">
              <div className="relative">
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-2 right-2 z-10 h-8 w-8 p-0"
                  onClick={() => copyCode(code, value)}
                >
                  {copiedTab === value ? (
                    <Check className="w-4 h-4 text-success" />
                  ) : (
                    <Copy className="w-4 h-4 text-muted-foreground" />
                  )}
                </Button>
                <ScrollArea className="h-[280px] rounded-lg bg-background/50 border border-border">
                  <pre className="p-4 text-xs font-mono text-muted-foreground overflow-x-auto">
                    <code>{code}</code>
                  </pre>
                </ScrollArea>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
};
