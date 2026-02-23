import type { ModelWeights } from '../types';

export interface AlexandreContext {
  /** Gradient norms per client for current round (clientId -> norm) */
  gradientNorms: Record<string, number>;
  /** Cosine similarity per client for current round (clientId -> similarity) */
  cosineSimilarities: Record<string, number>;
  /** Available cluster models */
  clusterModels: ModelWeights[];
  /** Global model as fallback */
  globalModel: ModelWeights;
}

/**
 * Assigns a model to a client based on gradient norms and cosine similarities.
 * 
 * @param clientId - The client to assign a model to
 * @param context - Contains gradientNorms and cosineSimilarities for all clients
 * @returns The model weights to send to this client
 */
export function getModelForAlexandre(
  clientId: string,
  context: AlexandreContext
): ModelWeights {
  const _gradientNorm = context.gradientNorms[clientId];
  const _cosineSimilarity = context.cosineSimilarities[clientId];

  // TODO: Implement assignment logic using gradientNorm and cosineSimilarity
  // Available data:
  //   - _gradientNorm: L2 norm of the client's gradient (delta weights between round N-1 and N)
  //   - _cosineSimilarity: cosine similarity of the client's delta weights with its cluster members
  //   - context.clusterModels: array of cluster model weights
  //   - context.globalModel: the global aggregated model
  //
  // Return the ModelWeights to assign to this client for the next round.

  return context.globalModel;
}
