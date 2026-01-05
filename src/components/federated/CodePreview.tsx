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

  const clientCode = `// MLP réel pour XOR - Entraînement client
import { trainEpoch, computeAccuracy, cloneWeights } from './mlp';

// Chaque client reçoit le modèle global et l'entraîne localement
const trainClient = async (globalMLP: MLPWeights, clientData) => {
  const localMLP = cloneWeights(globalMLP);
  
  const { inputs, outputs } = clientData;
  const localEpochs = 5;
  const learningRate = 0.5;
  
  let loss = 0;
  let accuracy = 0;
  
  for (let epoch = 0; epoch < localEpochs; epoch++) {
    loss = trainEpoch(inputs, outputs, localMLP, learningRate);
    accuracy = computeAccuracy(inputs, outputs, localMLP);
  }
  
  return { weights: localMLP, loss, accuracy };
};`;

  const aggregationCode = `// Agrégation FedAVG réelle
const fedAvg = (clientWeights) => {
  const totalDataSize = clientWeights.reduce((sum, c) => sum + c.dataSize, 0);
  
  // Moyenne pondérée par taille des données
  const aggregatedW1 = initZeroMatrix(2, 8);
  const aggregatedW2 = initZeroMatrix(8, 1);
  
  for (const { weights, dataSize } of clientWeights) {
    const weight = dataSize / totalDataSize;
    
    // W1: [2 x 8], W2: [8 x 1]
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 8; j++) {
        aggregatedW1[i][j] += weights.W1[i][j] * weight;
      }
    }
    for (let i = 0; i < 8; i++) {
      aggregatedW2[i][0] += weights.W2[i][0] * weight;
    }
  }
  
  return { W1: aggregatedW1, W2: aggregatedW2, b1, b2 };
};`;

  const serverCode = `// Génération des données XOR synthétiques
const generateClientData = (numSamples: number, noiseLevel = 0.1) => {
  const inputs = [];
  const outputs = [];
  
  for (let i = 0; i < numSamples; i++) {
    const x1 = Math.random() > 0.5 ? 1 : 0;
    const x2 = Math.random() > 0.5 ? 1 : 0;
    const xorResult = x1 !== x2 ? 1 : 0;
    
    inputs.push([
      x1 + (Math.random() - 0.5) * noiseLevel,
      x2 + (Math.random() - 0.5) * noiseLevel,
    ]);
    outputs.push([xorResult]);
  }
  return { inputs, outputs };
};

// Architecture MLP: 2 → 8 → 1 (ReLU + Sigmoid)
const mlp = initializeMLPWeights(2, 8, 1);`;

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
