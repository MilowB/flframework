// Model Assignment Strategy: Cosine Similarity Drop Detection
// Detects sudden drops in cosine similarity and reassigns clients to the closest cluster

import type { ModelWeights, ClientState } from '../core/types';
import { clusterModelStore } from '../core/stores';
import { getModelFor1NN } from './oneNN';

// Store for tracking cosine similarity history per client
// Key: clientId, Value: array of cosine similarity values per round
export const clientCosineSimilarityHistory: Map<string, number[]> = new Map();

// Store for clients that need reassignment at next round
// Key: clientId, Value: target cluster index
export const clientReassignmentStore: Map<string, number> = new Map();

// Threshold for detecting a significant drop (40% lower)
const DROP_THRESHOLD = 0.25;

// Reset stores (for clean experiment restart)
export const resetCosineSimilarityStores = (): void => {
  clientCosineSimilarityHistory.clear();
  clientReassignmentStore.clear();
};

// Record cosine similarity for a client at a given round
export const recordClientCosineSimilarity = (
  clientId: string,
  cosineSimilarity: number,
  round: number
): void => {
  const history = clientCosineSimilarityHistory.get(clientId) || [];
  // Ensure we have the right index for the round
  while (history.length < round) {
    history.push(NaN); // Fill gaps with NaN
  }
  history[round] = cosineSimilarity;
  clientCosineSimilarityHistory.set(clientId, history);
};

// Detect if a client had a significant drop in cosine similarity
// Returns true if similarity at round N is 40% lower than round N-1
export const detectCosineSimilarityDrop = (
  clientId: string,
  currentRound: number
): boolean => {
  const history = clientCosineSimilarityHistory.get(clientId);
  if (!history || currentRound < 2) return false;
  
  const currentSim = history[currentRound - 1]; // Round N (just completed)
  const previousSim = history[currentRound - 2]; // Round N-1
  
  if (isNaN(currentSim) || isNaN(previousSim) || previousSim <= 0) return false;
  
  // Calculate drop percentage
  const dropRatio = (previousSim - currentSim) / Math.abs(previousSim);
  console.log(`previousSim ${previousSim}, currentSim ${currentSim}`)
  console.log("client " + clientId + " - dropRatio : " + dropRatio);
  return dropRatio >= DROP_THRESHOLD;
};

// Find the cluster that the client is most similar to based on distance matrix
export const findClosestCluster = (
  clientId: string,
  distanceMatrix: number[][],
  clusters: string[][],
  participatingClients: string[]
): number => {
  // Find client index in the distance matrix
  const clientIdx = participatingClients.indexOf(clientId);
  if (clientIdx === -1) return -1;
  
  // Find which cluster this client currently belongs to
  let currentCluster = -1;
  for (let c = 0; c < clusters.length; c++) {
    if (clusters[c].includes(clientId)) {
      currentCluster = c;
      break;
    }
  }

  // Calculate average distance to each cluster (excluding current)
  let minAvgDistance = Infinity;
  let closestCluster = -1;
  
  for (let c = 0; c < clusters.length; c++) {
    if (c === currentCluster) continue; // Skip current cluster
    
    const clusterMembers = clusters[c];
    if (clusterMembers.length === 0) continue;
    
    let totalDistance = 0;
    let count = 0;
    
    for (const memberId of clusterMembers) {
      const memberIdx = participatingClients.indexOf(memberId);
      if (memberIdx === -1 || memberIdx === clientIdx) continue;
      
      totalDistance += distanceMatrix[clientIdx][memberIdx];
      count++;
    }
    
    if (count > 0) {
      const avgDistance = totalDistance / count;
      if (avgDistance < minAvgDistance) {
        minAvgDistance = avgDistance;
        closestCluster = c;
      }
    }
  }
  
  return closestCluster;
};

// Find which clients the given client has moved closer to between two rounds
// Returns an array of { clientId, distanceChange, newDistance } sorted by most approached (largest negative change)
export const findApproachedClients = (
  clientId: string,
  currentDistanceMatrix: number[][],
  previousDistanceMatrix: number[][],
  clusters: string[][],
  participatingClients: string[]
): { clientId: string; distanceChange: number; newDistance: number; clusterId: number }[] => {
  // Find client index in the distance matrix
  const clientIdx = participatingClients.indexOf(clientId);
  if (clientIdx === -1) return [];
  
  // Validate matrix dimensions
  if (!currentDistanceMatrix || !previousDistanceMatrix) return [];
  if (currentDistanceMatrix.length !== previousDistanceMatrix.length) return [];
  if (clientIdx >= currentDistanceMatrix.length || clientIdx >= previousDistanceMatrix.length) return [];
  
  const results: { clientId: string; distanceChange: number; newDistance: number; clusterId: number }[] = [];
  
  // Compare distances to all other clients
  for (let otherIdx = 0; otherIdx < participatingClients.length; otherIdx++) {
    if (otherIdx === clientIdx) continue; // Skip self
    
    const otherId = participatingClients[otherIdx];
    
    // Get distances from both matrices
    const currentDistance = currentDistanceMatrix[clientIdx]?.[otherIdx];
    const previousDistance = previousDistanceMatrix[clientIdx]?.[otherIdx];
    
    // Skip if either distance is undefined or NaN
    if (currentDistance === undefined || previousDistance === undefined) continue;
    if (isNaN(currentDistance) || isNaN(previousDistance)) continue;
    
    // Calculate distance change (negative = got closer)
    const distanceChange = currentDistance - previousDistance;
    
    // Find which cluster this other client belongs to
    let clusterId = -1;
    for (let c = 0; c < clusters.length; c++) {
      if (clusters[c].includes(otherId)) {
        clusterId = c;
        break;
      }
    }
    
    results.push({
      clientId: otherId,
      distanceChange,
      newDistance: currentDistance,
      clusterId
    });
  }
  
  // Sort by distance change (most negative first = most approached)
  results.sort((a, b) => a.distanceChange - b.distanceChange);
  
  return results;
};

// Find which cluster the client has moved closest to (based on average distance change)
export const findMostApproachedCluster = (
  clientId: string,
  currentDistanceMatrix: number[][],
  previousDistanceMatrix: number[][],
  clusters: string[][],
  participatingClients: string[]
): { clusterId: number; avgDistanceChange: number; approachedClients: string[] } | null => {
  const approachedClients = findApproachedClients(
    clientId,
    currentDistanceMatrix,
    previousDistanceMatrix,
    clusters,
    participatingClients
  );
  
  if (approachedClients.length === 0) return null;
  
  // Find current cluster of the client
  let currentCluster = -1;
  for (let c = 0; c < clusters.length; c++) {
    if (clusters[c].includes(clientId)) {
      currentCluster = c;
      break;
    }
  }
  
  // Calculate average distance change per cluster (excluding current cluster)
  const clusterChanges: Map<number, { totalChange: number; count: number; clients: string[] }> = new Map();
  
  for (const result of approachedClients) {
    if (result.clusterId === -1 || result.clusterId === currentCluster) continue;
    
    const existing = clusterChanges.get(result.clusterId) || { totalChange: 0, count: 0, clients: [] };
    existing.totalChange += result.distanceChange;
    existing.count++;
    if (result.distanceChange < 0) { // Only add if actually approached
      existing.clients.push(result.clientId);
    }
    clusterChanges.set(result.clusterId, existing);
  }
  
  // Find cluster with most negative average change
  let bestCluster = -1;
  let bestAvgChange = 0;
  let bestClients: string[] = [];
  
  for (const [clusterId, data] of clusterChanges) {
    const avgChange = data.totalChange / data.count;
    if (avgChange < bestAvgChange) {
      bestAvgChange = avgChange;
      bestCluster = clusterId;
      bestClients = data.clients;
    }
  }
  
  if (bestCluster === -1) return null;
  
  return {
    clusterId: bestCluster,
    avgDistanceChange: bestAvgChange,
    approachedClients: bestClients
  };
};

// Schedule a client for reassignment at the next round
export const scheduleReassignment = (
  clientId: string,
  targetCluster: number
): void => {
  if (targetCluster >= 0) {
    clientReassignmentStore.set(clientId, targetCluster);
  }
};

// Check if a client is scheduled for reassignment
export const hasScheduledReassignment = (clientId: string): boolean => {
  return clientReassignmentStore.has(clientId);
};

// Get the target cluster for a scheduled reassignment
export const getScheduledReassignment = (clientId: string): number | undefined => {
  return clientReassignmentStore.get(clientId);
};

// Clear reassignment after it's been applied
export const clearReassignment = (clientId: string): void => {
  clientReassignmentStore.delete(clientId);
};

// Get model for Cosine Similarity assignment strategy
export const getModelForCosineSimilarity = (
  clientId: string,
  globalModel: ModelWeights
): ModelWeights => {
  // Check if this client has a scheduled reassignment
  const targetCluster = clientReassignmentStore.get(clientId);
  
  if (targetCluster !== undefined) {
    // Get the model for the target cluster
    const clusterModel = clusterModelStore.get(`cluster-${targetCluster}`);
    if (clusterModel) {
      // Clear the reassignment after applying it
      clientReassignmentStore.delete(clientId);
      return clusterModel;
    }
  }
  
  // Default: use 1NN (return the cluster model the client belongs to)
  return getModelFor1NN(clientId, globalModel);
};
