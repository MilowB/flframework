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

  const clientCode = `// MLP pour MNIST - Entraînement client
import { trainEpoch, computeAccuracy, cloneWeights } from './mlp';
import { loadMNISTTrain, getClientDataSubset } from './mnist';

// Chaque client reçoit le modèle global et l'entraîne localement
const trainClient = async (globalMLP: MLPWeights, clientId: string) => {
  const localMLP = cloneWeights(globalMLP);
  
  // Données MNIST non-IID pour ce client
  const mnist = await loadMNISTTrain();
  const { inputs, outputs } = getClientDataSubset(mnist, clientId, 400, true);
  
  const localEpochs = 3;
  const learningRate = 0.01;
  
  for (let epoch = 0; epoch < localEpochs; epoch++) {
    const loss = trainEpoch(inputs, outputs, localMLP, learningRate);
    const accuracy = computeAccuracy(inputs, outputs, localMLP);
    console.log(\`Epoch \${epoch + 1}: loss=\${loss.toFixed(4)}, acc=\${accuracy.toFixed(2)}\`);
  }
  
  return { weights: localMLP };
};`;

  const aggregationCode = `// Agrégation FedAVG pour MNIST (784 → 128 → 10)
const fedAvg = (clientWeights) => {
  const totalDataSize = clientWeights.reduce((sum, c) => sum + c.dataSize, 0);
  
  // Moyenne pondérée par taille des données
  const aggregated = {
    W1: zeroMatrix(784, 128), // Input → Hidden
    b1: zeros(128),
    W2: zeroMatrix(128, 10),  // Hidden → Output (10 classes)
    b2: zeros(10),
  };
  
  for (const { weights, dataSize } of clientWeights) {
    const w = dataSize / totalDataSize;
    
    for (let i = 0; i < 784; i++)
      for (let j = 0; j < 128; j++)
        aggregated.W1[i][j] += weights.W1[i][j] * w;
    
    for (let i = 0; i < 128; i++)
      for (let j = 0; j < 10; j++)
        aggregated.W2[i][j] += weights.W2[i][j] * w;
  }
  
  return aggregated;
};`;

  const serverCode = `// Chargement MNIST et distribution non-IID
import { loadMNISTTrain, getClientDataSubset, oneHot } from './mnist';

// Charger le dataset (60,000 images train, 10,000 test)
const mnist = await loadMNISTTrain();
// → { images: number[60000][784], labels: number[60000] }

// Distribution non-IID: chaque client spécialisé sur 2-3 chiffres
const clientData = getClientDataSubset(mnist, 'client-0', 400, true);
// → { inputs: number[400][784], outputs: number[400][10] }

// Architecture MLP: 784 → 128 → 10 (ReLU + Softmax)
const mlp = initializeMLPWeights(784, 128, 10);

// Forward pass avec softmax pour classification
const { output } = forward(image, mlp);
const predictedDigit = output.indexOf(Math.max(...output));`;


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
