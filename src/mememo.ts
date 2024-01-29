/**
 * Mememo
 * @author: Jay Wang (jay@zijie.wang)
 */

import { randomLcg, randomUniform } from 'd3-random';
import { MinHeap, MaxHeap, IGetCompareValue } from '@datastructures-js/heap';

export const add = (a: number, b: number) => {
  return a + b;
};

type BuiltInDistanceFunction = 'cosine' | 'cosine-normalized';

interface SearchNodeCandidate<T> {
  key: T;
  distance: number;
}

/**
 * - 'cosine': Cosine distance
 * - 'cosine-normalized': Cosine distance between two normalized vectors
 */
type DistanceFunction =
  | BuiltInDistanceFunction
  | ((a: number[], b: number[]) => number);

// Built-in distance functions
const DISTANCE_FUNCTIONS: Record<
  BuiltInDistanceFunction,
  (a: number[], b: number[]) => number
> = {
  cosine: (a: number[], b: number[]) => {
    const dotProduct = a.reduce(
      (sum, value, index) => sum + value * b[index],
      0
    );
    const magnitudeA = Math.sqrt(a.reduce((sum, value) => sum + value ** 2, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, value) => sum + value ** 2, 0));
    return 1 - dotProduct / (magnitudeA * magnitudeB);
  },

  'cosine-normalized': (a: number[], b: number[]) => {
    const dotProduct = a.reduce(
      (sum, value, index) => sum + value * b[index],
      0
    );
    return 1 - dotProduct;
  }
};

interface HNSWConfig {
  /** Distance function. */
  distanceFunction?: DistanceFunction;

  /** The max number of neighbors for each node. A reasonable range of m is from
   * 5 to 48. Smaller m generally produces better results for lower recalls
   * and/or lower dimensional data, while bigger m is better for high recall
   * and/or high dimensional data. */
  m?: number;

  /** The number of neighbors to consider in construction's greedy search. */
  efConstruction?: number;

  /** The number of neighbors to keep for each node at the first level. */
  mMax0?: number;

  /** Normalizer parameter controlling number of overlaps across layers. */
  ml?: number;

  /** Optional random seed. */
  seed?: number;
}

/**
 * A node in the HNSW graph.
 */
class Node<T> {
  /** The unique key of an element. */
  key: T;

  /** The embedding value of the element. */
  value: number[];

  constructor(key: T, value: number[]) {
    this.key = key;
    this.value = value;
  }
}

/**
 * One graph layer in the HNSW index
 */
class GraphLayer<T> {
  /** The graph maps a key to its neighbor and distances */
  graph: Map<T, Map<T, number>>;

  /**
   * Initialize a new graph layer.
   * @param key The first key to insert into the graph layer.
   */
  constructor(key: T) {
    this.graph = new Map<T, Map<T, number>>();
    this.graph.set(key, new Map<T, number>());
  }
}

/**
 * HNSW (Hierarchical Navigable Small World) class.
 */
export class HNSW<T = string> {
  distanceFunction: (a: number[], b: number[]) => number;

  /** The max number of neighbors for each node. */
  m: number;

  /** The number of neighbors to consider in construction's greedy search. */
  efConstruction: number;

  /** The number of neighbors to keep for each node at the first level. */
  mMax0: number;

  /** Normalizer parameter controlling number of overlaps across layers. */
  ml: number;

  /** Seeded random number generator */
  rng: () => number;

  /** A collection all the nodes */
  nodes: Map<T, Node<T>>;

  /** A list of all layers */
  graphLayers: GraphLayer<T>[];

  /** Current entry point of the graph */
  entryPointKey: T | null = null;

  /**
   * Constructs a new instance of the class.
   * @param config - The configuration object.
   * @param config.distanceFunction - Distance function. Default: 'cosine'
   * @param config.m -  The max number of neighbors for each node. A reasonable
   * range of m is from 5 to 48. Smaller m generally produces better results for
   * lower recalls and/or lower dimensional data, while bigger m is better for
   * high recall and/or high dimensional data. Default: 16
   * @param config.efConstruction - The number of neighbors to consider in
   * construction's greedy search. Default: 100
   * @param config.mMax0 - The maximum number of connections that a node can
   * have in the zero layer. Default 2 * m.
   * @param config.ml - Normalizer parameter. Default 1 / ln(m)
   * @param config.seed - Optional random seed.
   */
  constructor({
    distanceFunction,
    m,
    efConstruction,
    mMax0,
    ml,
    seed
  }: HNSWConfig) {
    // Initialize HNSW parameters
    this.m = m || 16;
    this.efConstruction = efConstruction || 100;
    this.mMax0 = mMax0 || this.m * 2;
    this.ml = ml || 1 / Math.log(this.m);

    if (seed) {
      this.rng = randomLcg(seed);
    } else {
      this.rng = randomLcg(randomUniform()());
    }

    // Set the distance function
    if (distanceFunction === undefined) {
      this.distanceFunction = DISTANCE_FUNCTIONS['cosine'];
    } else {
      if (typeof distanceFunction === 'string') {
        this.distanceFunction = DISTANCE_FUNCTIONS[distanceFunction];
      } else {
        this.distanceFunction = distanceFunction;
      }
    }

    // Data structures
    this.graphLayers = [];
    this.nodes = new Map<T, Node<T>>();
  }

  /**
   * Insert a new element to the index.
   * @param key Key of the new element.
   * @param value The embedding of the new element to insert.
   */
  insert(key: T, value: number[]) {
    // If the key already exists, update the node
    if (this.nodes.has(key)) {
      // TODO: Update the node
      return;
    }

    // Randomly determine the max level of this node
    const level = this._getRandomLevel();

    if (this.entryPointKey !== null) {
      // (1): Search closest point from layers above
      // Top layer => all layers above the new node's highest layer
      const entryPointInfo = this.nodes.get(this.entryPointKey);
      if (entryPointInfo === undefined) {
        throw Error(`Can't find node info of ${this.entryPointKey}`);
      }

      // Start with the entry point
      let minDistance = this.distanceFunction(value, entryPointInfo.value);
      let minNodeKey: T = this.entryPointKey;

      for (let l = this.graphLayers.length - 1; l >= level + 1; l--) {
        const result = this._searchLayerEF1(
          value,
          minNodeKey,
          minDistance,
          this.graphLayers[l]
        );
        minDistance = result.minDistance;
        minNodeKey = result.minNodeKey;
      }

      // (2): Insert the node
      // New node's highest layer => layer 0
    }

    // If the level is beyond current layers, extend the layers
    for (let l = this.graphLayers.length; l < level + 1; l++) {
      this.graphLayers.push(new GraphLayer(key));

      // Set entry point as the last added node
      this.entryPointKey = key;
    }
  }

  /**
   * Greedy search the closest neighbor in a layer.
   * @param queryValue The embedding value of the query
   * @param entryPointKey Current entry point of this layer
   * @param entryPointDistance Distance between query and entry point
   * @param graphLayer Current graph layer
   */
  _searchLayerEF1(
    queryValue: number[],
    entryPointKey: T,
    entryPointDistance: number,
    graphLayer: GraphLayer<T>
  ) {
    const nodeCandidateCompare: IGetCompareValue<SearchNodeCandidate<T>> = (
      candidate: SearchNodeCandidate<T>
    ) => candidate.distance;
    const candidateHeap = new MinHeap(nodeCandidateCompare);

    // Initialize the min heap with the current entry point
    candidateHeap.push({ key: entryPointKey, distance: entryPointDistance });

    // Find the node with the minimal distance using greedy graph search
    let minNodeKey = entryPointKey;
    let minDistance = entryPointDistance;
    const visitedNodes = new Set<T>();

    while (candidateHeap.size() > 0) {
      const curCandidate = candidateHeap.pop()!;
      if (curCandidate.distance > minDistance) {
        break;
      }

      const curNode = graphLayer.graph.get(curCandidate.key);
      if (curNode === undefined) {
        throw Error(`Cannot find node with key ${curCandidate.key}`);
      }

      for (const key of curNode.keys()) {
        if (!visitedNodes.has(key)) {
          visitedNodes.add(key);
          // Compute the distance between the node and query
          const curNodeInfo = this.nodes.get(key);
          if (curNodeInfo === undefined) {
            throw Error(`Cannot find node info with key ${key}`);
          }
          const distance = this.distanceFunction(curNodeInfo.value, queryValue);

          // Continue explore the node's neighbors if the distance is improving
          if (distance < minDistance) {
            minDistance = distance;
            minNodeKey = key;
            candidateHeap.push({ key, distance });
          }
        }
      }
    }

    return {
      minNodeKey,
      minDistance
    };
  }

  /**
   * Greedy search `ef` closest points in a given layer
   * @param queryValue Embedding value of the query point
   * @param entryPoints Entry points of this layer
   * @param graphLayer Current layer to search
   * @param ef Number of neighbors to consider during search
   */
  _searchLayer(
    queryValue: number[],
    entryPoints: SearchNodeCandidate<T>[],
    graphLayer: GraphLayer<T>,
    ef: number
  ) {
    // We maintain two heaps in this function
    // For candidate nodes, we use a min heap to get the closest node
    // For found nearest nodes, we use a max heap to get the furthest node
    const nodeCompare: IGetCompareValue<SearchNodeCandidate<T>> = (
      candidate: SearchNodeCandidate<T>
    ) => candidate.distance;

    const candidateMinHeap = new MinHeap(nodeCompare);
    const foundNodesMaxHeap = new MaxHeap(nodeCompare);
    const visitedNodes = new Set<T>();

    for (const searchNode of entryPoints) {
      candidateMinHeap.push(searchNode);
      foundNodesMaxHeap.push(searchNode);
      visitedNodes.add(searchNode.key);
    }

    while (candidateMinHeap.size() > 0) {
      const nearestCandidate = candidateMinHeap.pop()!;
      const furthestFoundNode = foundNodesMaxHeap.root()!;

      if (nearestCandidate.distance > furthestFoundNode.distance) {
        break;
      }

      // Update candidates and found nodes using the current node's neighbors
      const curNode = graphLayer.graph.get(nearestCandidate.key);
      if (curNode === undefined) {
        throw Error(`Cannot find node with key ${nearestCandidate.key}`);
      }

      for (const neighborKey of curNode.keys()) {
        if (!visitedNodes.has(neighborKey)) {
          visitedNodes.add(neighborKey);

          // Compute the distance of the neighbor and query
          const neighborInfo = this.nodes.get(neighborKey);
          if (neighborInfo === undefined) {
            throw Error(`Cannot find node with key ${neighborKey}`);
          }
          const distance = this.distanceFunction(
            queryValue,
            neighborInfo.value
          );
          const furthestFoundNode = foundNodesMaxHeap.root()!;

          // Add this node if it is better than our found nodes or we do not
          // have enough found nodes
          if (
            distance < furthestFoundNode.distance ||
            foundNodesMaxHeap.size() < ef
          ) {
            candidateMinHeap.push({ key: neighborKey, distance });
            foundNodesMaxHeap.push({ key: neighborKey, distance });

            // If we have more found nodes than ef, remove the furthest point
            if (foundNodesMaxHeap.size() > ef) {
              foundNodesMaxHeap.pop();
            }
          }
        }
      }
    }

    return foundNodesMaxHeap.toArray();
  }

  /**
   * Generate a random level for a node using a exponentially decaying
   * probability distribution
   */
  _getRandomLevel() {
    return Math.floor(-Math.log(this.rng()) * this.ml);
  }
}
