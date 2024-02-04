/**
 * Mememo
 * @author: Jay Wang (jay@zijie.wang)
 */

import { randomLcg, randomUniform } from 'd3-random';
import { MinHeap, MaxHeap, IGetCompareValue } from '@datastructures-js/heap';
import { openDB, IDBPDatabase } from 'idb';

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

  /** Whether the node is marked as deleted. */
  isDeleted: boolean;

  constructor(key: T, value: number[]) {
    this.key = key;
    this.value = value;
    this.isDeleted = false;
  }
}

/**
 * An abstraction of a map storing nodes (in memory or in indexedDB)
 */
class Nodes<T> {
  nodesMap: Map<T, Node<T>>;
  dbPromise: Promise<IDBPDatabase<string>> | null;

  constructor(indexedDBStoreName?: string) {
    this.nodesMap = new Map<T, Node<T>>();
    this.dbPromise = null;

    if (indexedDBStoreName !== undefined) {
      // Create a new store
      this.dbPromise = openDB<string>('mememo-index-store', 1, {
        upgrade(db) {
          db.createObjectStore(indexedDBStoreName);
        }
      }).then(async db => {
        // Clear the store from previous sessions
        const tx = db.transaction(indexedDBStoreName, 'readwrite');
        const store = tx.objectStore(indexedDBStoreName);
        await store.clear();
        await tx.done;
        return db;
      });
    }
  }

  async size() {
    if (this.dbPromise === null) {
      return this.nodesMap.size;
    } else {
      return 1;
    }
  }

  async has(key: T) {
    if (this.dbPromise === null) {
      return this.nodesMap.has(key);
    } else {
      return false;
    }
  }

  async get(key: T) {
    if (this.dbPromise === null) {
      return this.nodesMap.get(key);
    } else {
      return undefined;
    }
  }

  async set(key: T, value: Node<T>) {
    if (this.dbPromise === null) {
      this.nodesMap.set(key, value);
    } else {
      // pass
    }
  }

  async clear() {
    if (this.dbPromise === null) {
      this.nodesMap = new Map<T, Node<T>>();
    } else {
      // pass
    }
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
  nodes: Nodes<T>;

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
    this.nodes = new Nodes();
  }

  /**
   * Insert a new element to the index.
   * @param key Key of the new element.
   * @param value The embedding of the new element to insert.
   * @param maxLevel The max layer to insert this element. You don't need to set
   * this value in most cases. We add this parameter for testing purpose.
   */
  async insert(key: T, value: number[], maxLevel?: number | undefined) {
    // If the key already exists, throw an error
    if (await this.nodes.has(key)) {
      const nodeInfo = await this._getNodeInfo(key);

      if (nodeInfo.isDeleted) {
        // The node was flagged as deleted, so we update it using the new value
        nodeInfo.isDeleted = false;
        await this.nodes.set(key, nodeInfo);
        await this.update(key, value);
        return;
      }

      throw Error(
        `There is already a node with key ${key} in the index. ` +
          'Use update() to update this node.'
      );
    }

    // Randomly determine the max level of this node
    const level = maxLevel === undefined ? this._getRandomLevel() : maxLevel;
    // console.log('random level:', level);

    // Add this node to the node index first
    await this.nodes.set(key, new Node(key, value));

    if (this.entryPointKey !== null) {
      // (1): Search closest point from layers above
      const entryPointInfo = await this._getNodeInfo(this.entryPointKey);

      // Start with the entry point
      let minDistance = this.distanceFunction(value, entryPointInfo.value);
      let minNodeKey: T = this.entryPointKey;

      // Top layer => all layers above the new node's highest layer
      for (let l = this.graphLayers.length - 1; l >= level + 1; l--) {
        const result = await this._searchLayerEF1(
          value,
          minNodeKey,
          minDistance,
          this.graphLayers[l]
        );
        minDistance = result.minDistance;
        minNodeKey = result.minNodeKey;
      }

      // (2): Insert the node from its random layer to layer 0
      let entryPoints: SearchNodeCandidate<T>[] = [
        { key: minNodeKey, distance: minDistance }
      ];

      // New node's highest layer => layer 0
      const nodeHightLayerLevel = Math.min(this.graphLayers.length - 1, level);
      for (let l = nodeHightLayerLevel; l >= 0; l--) {
        // Layer 0 could have a different neighbor size constraint
        const levelM = l === 0 ? this.mMax0 : this.m;

        // Search for closest points at this level to connect with
        entryPoints = await this._searchLayer(
          value,
          entryPoints,
          this.graphLayers[l],
          this.efConstruction
        );

        // Prune the neighbors so we have at most levelM neighbors
        const selectedNeighbors = await this._selectNeighborsHeuristic(
          entryPoints,
          levelM
        );

        // Insert the new node
        const newNode = new Map<T, number>();
        for (const neighbor of selectedNeighbors) {
          newNode.set(neighbor.key, neighbor.distance);
        }
        this.graphLayers[l].graph.set(key, newNode);

        // We also need to update this new node's neighbors so that their
        // neighborhood include this new node
        for (const neighbor of selectedNeighbors) {
          const neighborNode = this.graphLayers[l].graph.get(neighbor.key);
          if (neighborNode === undefined) {
            throw Error(`Can't find neighbor node ${neighbor.key}`);
          }

          // Add the neighbor's existing neighbors as candidates
          const neighborNeighborCandidates: SearchNodeCandidate<T>[] = [];
          for (const [key, distance] of neighborNode.entries()) {
            const candidate: SearchNodeCandidate<T> = { key, distance };
            neighborNeighborCandidates.push(candidate);
          }

          // Add the new node as a candidate as well
          neighborNeighborCandidates.push({ key, distance: neighbor.distance });

          // Apply the same heuristic to prune the neighbor's neighbors
          const selectedNeighborNeighbors =
            await this._selectNeighborsHeuristic(
              neighborNeighborCandidates,
              levelM
            );

          // Update this neighbor's neighborhood
          const newNeighborNode = new Map<T, number>();
          for (const neighborNeighbor of selectedNeighborNeighbors) {
            newNeighborNode.set(
              neighborNeighbor.key,
              neighborNeighbor.distance
            );
          }
          this.graphLayers[l].graph.set(neighbor.key, newNeighborNode);
        }
      }
    }

    // If the level is beyond current layers, extend the layers
    for (let l = this.graphLayers.length; l < level + 1; l++) {
      this.graphLayers.push(new GraphLayer(key));

      // Set entry point as the last added node
      this.entryPointKey = key;
    }
  }

  /**
   * Update an element in the index
   * @param key Key of the element.
   * @param value The new embedding of the element
   */
  async update(key: T, value: number[]) {
    if (!(await this.nodes.has(key))) {
      throw Error(
        `The node with key ${key} does not exist. ` +
          'Use insert() to add new node.'
      );
    }

    await this.nodes.set(key, new Node(key, value));

    if (this.entryPointKey === key && (await this.nodes.size()) === 1) {
      return;
    }

    // Re-index all the neighbors of this node in all layers
    for (let l = 0; l < this.graphLayers.length; l++) {
      const curGraphLayer = this.graphLayers[l];
      // Layer 0 could have a different neighbor size constraint
      const levelM = l === 0 ? this.mMax0 : this.m;

      // If the current layer doesn't have this node, then the upper layers
      // won't have it either
      if (!curGraphLayer.graph.has(key)) {
        break;
      }
      const curNode = curGraphLayer.graph.get(key)!;

      // For each neighbor, we use the entire second-degree neighborhood of the
      // updating node as new connection candidates
      const secondDegreeNeighborhood: Set<T> = new Set([key]);

      // Find the second-degree neighborhood
      for (const firstDegreeNeighbor of curNode.keys()) {
        secondDegreeNeighborhood.add(firstDegreeNeighbor);

        const firstDegreeNeighborNode =
          curGraphLayer.graph.get(firstDegreeNeighbor);
        if (firstDegreeNeighborNode === undefined) {
          throw Error(`Can't find node with key ${firstDegreeNeighbor}`);
        }

        for (const secondDegreeNeighbor of firstDegreeNeighborNode.keys()) {
          secondDegreeNeighborhood.add(secondDegreeNeighbor);
        }
      }

      // Update the first-degree neighbor's connections
      const nodeCompare: IGetCompareValue<SearchNodeCandidate<T>> = (
        candidate: SearchNodeCandidate<T>
      ) => candidate.distance;

      for (const firstDegreeNeighbor of curNode.keys()) {
        // (1) Find `efConstruction` number of candidates
        const candidateMaxHeap = new MaxHeap(nodeCompare);
        const firstDegreeNeighborInfo =
          await this._getNodeInfo(firstDegreeNeighbor);

        for (const secondDegreeNeighbor of secondDegreeNeighborhood) {
          if (secondDegreeNeighbor === firstDegreeNeighbor) {
            continue;
          }

          const secondDegreeNeighborInfo =
            await this._getNodeInfo(secondDegreeNeighbor);

          const distance = this.distanceFunction(
            firstDegreeNeighborInfo.value,
            secondDegreeNeighborInfo.value
          );

          if (candidateMaxHeap.size() < this.efConstruction) {
            // Add to the candidates if we still have open slots
            candidateMaxHeap.push({ key: secondDegreeNeighbor, distance });
          } else {
            // Add to the candidates if the distance is better than the worst
            // added candidate, by replacing the worst added candidate
            if (distance < candidateMaxHeap.top()!.distance) {
              candidateMaxHeap.pop();
              candidateMaxHeap.push({ key: secondDegreeNeighbor, distance });
            }
          }
        }

        // (2) Select `levelM` number candidates out of the candidates
        const candidates = candidateMaxHeap.toArray();
        const selectedCandidates = await this._selectNeighborsHeuristic(
          candidates,
          levelM
        );

        // (3) Update the neighbor's neighborhood
        const newNeighborNode = new Map<T, number>();
        for (const neighborNeighbor of selectedCandidates) {
          newNeighborNode.set(neighborNeighbor.key, neighborNeighbor.distance);
        }
        curGraphLayer.graph.set(firstDegreeNeighbor, newNeighborNode);
      }
    }

    // After re-indexing the neighbors of the updating node, we also need to
    // update the outgoing edges of the updating node in all layers. This is
    // similar to the initial indexing procedure in insert()
    await this._reIndexNode(key, value);
  }

  /**
   * Mark an element in the index as deleted.
   * This function does not delete the node from memory, but just remove it from
   * query result in the future. Future queries can still use this node to reach
   * other nodes. Future insertions will not add new edge to this node.
   *
   * See https://github.com/nmslib/hnswlib/issues/4 for discussion on the
   * challenges of deleting items in HNSW
   *
   * @param key Key of the node to delete
   */
  async markDeleted(key: T) {
    if (!(await this.nodes.has(key))) {
      throw Error(`Node with key ${key} does not exist.`);
    }

    // Special case: the user is trying to delete the entry point
    // We move the entry point to a neighbor or clean the entire graph if the
    // node is the last node
    if (this.entryPointKey === key) {
      let newEntryPointKey: T | null = null;
      // Traverse from top layer to layer 0
      for (let i = this.graphLayers.length - 1; i >= 0; i--) {
        for (const otherKey of this.graphLayers[i].graph.keys()) {
          const otherNode = await this.nodes.get(otherKey);
          if (otherKey !== key && !otherNode!.isDeleted) {
            newEntryPointKey = otherKey;
            break;
          }
        }

        if (newEntryPointKey !== null) {
          break;
        } else {
          // There is no more nodes in this layer, we can remove it.
          this.graphLayers.splice(i, 1);
        }
      }

      if (newEntryPointKey === null) {
        // There is no nodes in the index
        await this.clear();
        return;
      }

      this.entryPointKey = newEntryPointKey;
    }

    const nodeInfo = await this._getNodeInfo(key);
    nodeInfo.isDeleted = true;
    await this.nodes.set(key, nodeInfo);
  }

  /**
   * UnMark a deleted element in the index.
   *
   * See https://github.com/nmslib/hnswlib/issues/4 for discussion on the
   * challenges of deleting items in HNSW
   *
   * @param key Key of the node to recover
   */
  async unMarkDeleted(key: T) {
    const nodeInfo = await this._getNodeInfo(key);
    nodeInfo.isDeleted = false;
    await this.nodes.set(key, nodeInfo);
  }

  /**
   * Reset the index.
   */
  async clear() {
    this.graphLayers = [];
    await this.nodes.clear();
  }

  /**
   * Find k nearest neighbors of the query point
   * @param value Embedding value
   * @param k k nearest neighbors of the query value
   * @param ef Number of neighbors to search at each step
   */
  async query(
    value: number[],
    k: number | undefined = undefined,
    ef: number | undefined = this.efConstruction
  ) {
    if (this.entryPointKey === null) {
      throw Error('Index is not initialized yet');
    }

    // EF=1 search from the top layer to layer 1
    let minNodeKey: T = this.entryPointKey;
    const entryPointInfo = await this._getNodeInfo(minNodeKey);
    let minNodeDistance = this.distanceFunction(entryPointInfo.value, value);

    for (let l = this.graphLayers.length - 1; l >= 1; l--) {
      const result = await this._searchLayerEF1(
        value,
        minNodeKey,
        minNodeDistance,
        this.graphLayers[l],
        false
      );
      minNodeKey = result.minNodeKey;
      minNodeDistance = result.minDistance;
    }

    // EF search on layer 0
    const entryPoints: SearchNodeCandidate<T>[] = [
      { key: minNodeKey, distance: minNodeDistance }
    ];
    const candidates = await this._searchLayer(
      value,
      entryPoints,
      this.graphLayers[0],
      ef,
      false
    );

    candidates.sort((a, b) => a.distance - b.distance);

    if (k === undefined) {
      return candidates;
    } else {
      return candidates.slice(0, k);
    }
  }

  /**
   * Re-index an existing element's outgoing edges by repeating the insert()
   * algorithm (without updating its neighbor's edges)
   * @param key Key of an existing element
   * @param value Embedding value of an existing element
   */
  async _reIndexNode(key: T, value: number[]) {
    if (this.entryPointKey === null) {
      throw Error('entryPointKey is null');
    }

    let minNodeKey: T = this.entryPointKey;
    const entryPointInfo = await this._getNodeInfo(minNodeKey);
    let minNodeDistance = this.distanceFunction(entryPointInfo.value, value);
    let entryPoints: SearchNodeCandidate<T>[] = [
      { key: minNodeKey, distance: minNodeDistance }
    ];

    // Iterating through the top layer to layer 0
    // If the node is not in the layer => ef = 1 search
    // If the node is in the layer => ef search
    for (let l = this.graphLayers.length - 1; l >= 0; l--) {
      const curGraphLayer = this.graphLayers[l];

      if (!curGraphLayer.graph.has(key)) {
        // Layers above: Ef = 1 search
        const result = await this._searchLayerEF1(
          value,
          minNodeKey,
          minNodeDistance,
          curGraphLayer
        );
        minNodeKey = result.minNodeKey;
        minNodeDistance = result.minDistance;
      } else {
        // The node's top layer and layer below: EF search
        // Layer 0 could have a different neighbor size constraint
        const levelM = l === 0 ? this.mMax0 : this.m;

        // Search for closest points at this level to connect with
        entryPoints = await this._searchLayer(
          value,
          entryPoints,
          curGraphLayer,
          /** Here ef + 1 because this node is already in the index */
          this.efConstruction + 1
        );

        // Remove the current node itself as it would be selected (0 distance)
        entryPoints = entryPoints.filter(d => d.key !== key);

        // Prune the neighbors so we have at most levelM neighbors
        const selectedNeighbors = await this._selectNeighborsHeuristic(
          entryPoints,
          levelM
        );

        // Update the node's neighbors
        const newNode = new Map<T, number>();
        for (const neighbor of selectedNeighbors) {
          newNode.set(neighbor.key, neighbor.distance);
        }
        curGraphLayer.graph.set(key, newNode);
      }
    }
  }

  /**
   * Greedy search the closest neighbor in a layer.
   * @param queryValue The embedding value of the query
   * @param entryPointKey Current entry point of this layer
   * @param entryPointDistance Distance between query and entry point
   * @param graphLayer Current graph layer
   * @param canReturnDeletedNodes Whether to return deleted nodes
   */
  async _searchLayerEF1(
    queryValue: number[],
    entryPointKey: T,
    entryPointDistance: number,
    graphLayer: GraphLayer<T>,
    canReturnDeletedNode = true
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
          const curNodeInfo = await this._getNodeInfo(key);
          const distance = this.distanceFunction(curNodeInfo.value, queryValue);

          // Continue explore the node's neighbors if the distance is improving
          if (distance < minDistance) {
            // If the current node is marked as deleted, we do not return it as
            // a candidate, but we continue explore its neighbor
            if (!curNodeInfo.isDeleted || canReturnDeletedNode) {
              minDistance = distance;
              minNodeKey = key;
            }
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
   * @param canReturnDeletedNodes Whether to return deleted nodes
   */
  async _searchLayer(
    queryValue: number[],
    entryPoints: SearchNodeCandidate<T>[],
    graphLayer: GraphLayer<T>,
    ef: number,
    canReturnDeletedNodes = true
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
          const neighborInfo = await this._getNodeInfo(neighborKey);
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
            // If the current neighbor is marked as deleted, we do not return it
            // as a found node, but we continue explore its neighbor
            if (!neighborInfo.isDeleted || canReturnDeletedNodes) {
              foundNodesMaxHeap.push({ key: neighborKey, distance });
            }
            candidateMinHeap.push({ key: neighborKey, distance });

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
   * Simple heuristic to select neighbors. This function is different from
   * SELECT-NEIGHBORS-HEURISTIC in the HNSW paper. This function is based on
   * hnswlib and datasketch's implementations.
   * When selecting a neighbor, we compare the distance between selected
   * neighbors and the potential neighbor to the distance between the inserted
   * point and the potential neighbor. We favor neighbors that are further
   * away from selected neighbors to improve diversity.
   *
   * https://github.com/nmslib/hnswlib/blob/978f7137bc9555a1b61920f05d9d0d8252ca9169/hnswlib/hnswalg.h#L382
   * https://github.com/ekzhu/datasketch/blob/9973b09852a5018f23d831b1868da3a5d2ce6a3b/datasketch/hnsw.py#L832
   *
   * @param candidates Potential neighbors to select from
   * @param maxSize Max neighbors to connect to
   */
  async _selectNeighborsHeuristic(
    candidates: SearchNodeCandidate<T>[],
    maxSize: number
  ) {
    // candidates.length <= maxSize is more "correct", use < to be consistent
    // with other packages
    if (candidates.length < maxSize) {
      return candidates;
    }

    const nodeCompare: IGetCompareValue<SearchNodeCandidate<T>> = (
      candidate: SearchNodeCandidate<T>
    ) => candidate.distance;

    const candidateMinHeap = new MinHeap(nodeCompare);
    for (const candidate of candidates) {
      candidateMinHeap.insert(candidate);
    }

    const selectedNeighbors: SearchNodeCandidate<T>[] = [];

    while (candidateMinHeap.size() > 0) {
      if (selectedNeighbors.length >= maxSize) {
        return selectedNeighbors;
      }

      const candidate = candidateMinHeap.pop()!;
      let isCandidateFarFromExistingNeighbors = true;

      // Iterate selected neighbors to see if the candidate is further away
      for (const selectedNeighbor of selectedNeighbors) {
        const candidateInfo = await this._getNodeInfo(candidate.key);
        const neighborInfo = await this._getNodeInfo(selectedNeighbor.key);

        const distanceCandidateToNeighbor = this.distanceFunction(
          candidateInfo.value,
          neighborInfo.value
        );

        // Reject the candidate if
        // d(candidate, any approved candidate) < d(candidate, new node)
        if (distanceCandidateToNeighbor < candidate.distance) {
          isCandidateFarFromExistingNeighbors = false;
          break;
        }
      }

      if (isCandidateFarFromExistingNeighbors) {
        selectedNeighbors.push(candidate);
      }
    }

    return selectedNeighbors;
  }

  /**
   * Generate a random level for a node using a exponentially decaying
   * probability distribution
   */
  _getRandomLevel() {
    return Math.floor(-Math.log(this.rng()) * this.ml);
  }

  /**
   * Helper function to get the node in the global index
   * @param key Node key
   */
  async _getNodeInfo(key: T) {
    const node = await this.nodes.get(key);
    if (node === undefined) {
      throw Error(`Can't find node with key ${key}`);
    }
    return node;
  }
}
