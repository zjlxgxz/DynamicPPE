/* Copyright (c) 2021 --- ---

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

/**
 * DynamicPPE algorithm
 * To compile it:
 *      g++ -std=c++0x -O3 -g -Wall -o dynamic-ppe-fast \
            dynamic_ppe_fast.cpp MurmurHash3.cpp -lpthread
 *
 * DynamicPPE is built based on the following three papers:
 * [1]  Postăvaru, Ştefan, Anton Tsitsulin, Filipe Miguel Gonçalves de Almeida,
 *      Yingtao Tian, Silvio Lattanzi, and Bryan Perozzi. "InstantEmbedding:
 *      Efficient Local Node Representations." arXiv:2010.06992, 2020.
 * [2]  Zhang, Hongyang, Peter Lofgren, and Ashish Goel. "Approximate
 *      personalized pagerank on dynamic graphs." KDD, 2016.
 * [3]  Guo, Wentian, Yuchen Li, Mo Sha, and Kian-Lee Tan. "Parallel
 *      personalized pagerank on dynamic graphs." VLDB, 2017.
 */
#include <random>
#include <iostream>
#include <sstream>
#include <iterator>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "MurmurHash3.h"

using namespace std;
#define MAX_NODE_ID 6250000 // increase it if the number of nodes > 51200
#define MAX_STR_LEN 512
#define MAX_PRECISION 1e-06
typedef double d_type;
#define cpu_time(s_time) (double) (clock() - s_time) / CLOCKS_PER_SEC
#define wall_time(w_time) (double((chrono::steady_clock::now() - w_time)       \
                           .count())  * chrono::steady_clock::period::num /    \
                           chrono::steady_clock::period::den)

class dyn_lp {
public:
    int s;
    array<d_type, MAX_NODE_ID> p{};
    array<d_type, MAX_NODE_ID> r{};

    explicit dyn_lp(int s) {
        this->s = s;
    };

    d_type get_est() {
        d_type re = 0.0;
        for (auto it:this->p)
            re += it;
        return re;
    }

    d_type get_res() {
        d_type re = 0.0;
        for (auto it:this->r)
            re += it;
        return re;
    }
};

class ppr_worker {
public:
    uint32_t th_id;
    vector<int> target_nodes{};
    array<int, MAX_NODE_ID> pre_deg{};
    array<int, MAX_NODE_ID> q_posi{};
    array<int, MAX_NODE_ID> q_nega{};
    array<bool, MAX_NODE_ID> mark_posi{};
    array<bool, MAX_NODE_ID> mark_nega{};

    explicit ppr_worker(uint32_t thread_id) {
        this->th_id = thread_id;
    };
};

d_type *emb; // emb matrix with shape: num_snapshots * num_target_nodes * dim
double epsilon; // the coefficient of dynamic precision parameter
double alpha; // teleport, such as 0.15, 0.2.
// This random seed helps to choose two hash functions from MurmurHash3 family.
// We will also use this seed to generate a random embedding if one wants to
// have embedding vectors for singletons.
int seed;
int verbose;
// There are two possible values for model:
// 1) "hash": to use hash function; 2) "rand" to use random projection
char *model;
unordered_set<int> nodes;
size_t snapshot_index;
uint32_t num_cpus; // should be >= 1
uint64_t emb_dim; // embedding dimension, such as 128, 256, 512, 1024
uint64_t num_edges; // number of edges. It will +1 whenever a new edge comes
uint64_t num_snapshots; // total snapshots processed T
uint64_t emb_size; // num_snapshots * num_target_nodes * dim
uint64_t snapshot_size; // num_target_nodes * dim
vector<vector<int>> graph; // adjacency list of the input dynamic graph
// We save the hash values into arrays to boost the efficiency
array<uint32_t, MAX_NODE_ID> h1_arr{};
array<uint32_t, MAX_NODE_ID> h2_arr{};
d_type *rand_proj = nullptr;
uint64_t rand_proj_size;
array<int, MAX_NODE_ID> out_deg{}; // The degree distribution of G^t
array<int, MAX_NODE_ID> pre_deg{}; // The degree distribution of G^{t-1}
// The estimators p and residual r for all target_nodes.
unordered_map<int, dyn_lp *> estimators;
vector<tuple<int, int>> edges_batch; // Keep track of current \Delta G^t
unordered_map<int, ppr_worker *> workers; // A worker is a thread for p and r.
vector<int> work_nodes;
default_random_engine *generator;
uniform_real_distribution<> *uni_dist;
vector<int> emb_nodes;
array<int, MAX_NODE_ID> emb_nodes_indices; // Map the ids of emb_nodes
double eps_prime;
array<d_type, MAX_NODE_ID> eps_vec;
// paths of edge-list files
vector<string> list_g_paths;
char config_path[MAX_STR_LEN];
char re_emb_path[MAX_STR_LEN];
char re_stat_path[MAX_STR_LEN];

// run-time statistics
double run_time_io;
double run_time_push_cpu_time;
double run_time_push_wall_time;
double run_time_total_cpu_time;
double run_time_total_wall_time;


d_type gaussian_sample(d_type mean, d_type std) {
    // https://en.wikipedia.org/wiki/Marsaglia_polar_method
    static d_type spare;
    static bool has_spare = false;
    if (has_spare) {
        has_spare = false;
        return spare * std + mean;
    } else {
        d_type u, v, s;
        do {
            u = (*uni_dist)(*generator) * 2.0 - 1.0;
            v = (*uni_dist)(*generator) * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        has_spare = true;
        return mean + std * u * s;
    }
}

uint32_t rj_hash(uint32_t a) {
    // Robert Jenkins' 32 bit hash: https://gist.github.com/badboy/6267743
    a = (a + 0x7ed55d16) + (a << 12U);
    a = (a ^ 0xc761c23c) ^ (a >> 19U);
    a = (a + 0x165667b1) + (a << 5U);
    a = (a + 0xd3a2646c) ^ (a << 9U);
    a = (a + 0xfd7046c5) + (a << 3U);
    a = (a ^ 0xb55a4f09) ^ (a >> 16U);
    return a;
}

void batch_update_graph(const string &edge_path) {
    // update current graph g^{t-1} to g^t
    double start_time = clock();
    int uu, vv;
    ifstream infile(edge_path.c_str());
    if (infile.fail()) {
        cout << edge_path.c_str() << " does not exist (ignore it)! " << endl;
        return;
    }
    copy(begin(out_deg), end(out_deg), begin(pre_deg));
    edges_batch.clear();
    while (infile >> uu >> vv) {
        if (uu == vv) // ignore self-loops
            continue;
        nodes.insert(uu);
        nodes.insert(vv);
        //TODO: One should check whether (uu, vv) is a new edge or not.
        // We assume the input edges are all different from each other.
        graph[uu].push_back(vv);
        graph[vv].push_back(uu);
        out_deg[uu]++;
        out_deg[vv]++;
        if (snapshot_index != 0) {
            edges_batch.emplace_back(uu, vv);
            edges_batch.emplace_back(vv, uu);
        }
        num_edges++;
    }
    // adaptive precision parameter
    eps_prime = fmin(MAX_PRECISION, epsilon / (double) num_edges);
    for (int ii = 0; ii < MAX_NODE_ID; ii++)
        eps_vec[ii] = eps_prime * out_deg[ii];
    if (verbose == 1)
        cout << "finish read: " << edge_path
             << " in " << cpu_time(start_time) << " seconds" << endl;
    run_time_io += cpu_time(start_time);
}

void calculate_emb(int s) {
    // calculate embedding for current snapshot
    if (verbose == 1)
        cout << "calculate embedding for node: " << s << endl;
    uint64_t emb_bytes = sizeof(d_type) * emb_dim;
    uint64_t cur_snap_loc = snapshot_index * snapshot_size;
    uint64_t cur_emb_loc = emb_nodes_indices[s] * emb_dim;
    d_type *cur_emb = emb + cur_snap_loc + cur_emb_loc;
    memset(cur_emb, 0, emb_bytes);
    assert(estimators.find(s) != estimators.end());
    auto *lp = estimators[s];
    if (strcmp(model, "hash") == 0)
        for (int uu = 0; uu < MAX_NODE_ID; uu++) {
            if (lp->p[uu] == 0.0)
                continue;
            d_type p = fmax(log(lp->p[uu] * (d_type) nodes.size()), 0.0);
            if (p == 0.0)
                continue;
            cur_emb[h1_arr[uu]] += (h2_arr[uu] ? p : -p);
        }
    else
        for (int uu = 0; uu < MAX_NODE_ID; uu++) {
            if (lp->p[uu] == 0.0)
                continue;
            d_type p = fmax(log(lp->p[uu] * (d_type) nodes.size()), 0.0);
            if (p == 0.0)
                continue;
            // cblas_saxpy will be faster, use #include<cblas.h>
            d_type *proj_vec = rand_proj + uu * emb_dim;
            for (uint64_t ii = 0; ii < emb_dim; ii++) {
                cur_emb[ii] += p * proj_vec[ii];
            }
        }
}

static inline void *thread_local_forward_push_batch(void *th_para) {
    // update estimator and residual after seeing \Delta g^t
    int *worker_id = (int *) (th_para);
    ppr_worker *worker = workers[*worker_id];
    uint64_t num_pushes;
    int uu, edge_uu, edge_vv;
    d_type delta, res_uu, rest_prob;
    int front_posi, rear_posi;
    int front_nega, rear_nega;
    size_t nodes_size = sizeof(int) * MAX_NODE_ID;
    size_t mark_size = sizeof(bool) * MAX_NODE_ID;
    clock_t cpu_t_start = clock();
    auto wall_t_start = chrono::steady_clock::now();
    for (int s:worker->target_nodes) {
        if (estimators.find(s) == estimators.end())
            continue;
        auto *dlp = estimators[s];
        memcpy(&worker->pre_deg, &pre_deg, nodes_size);
        num_pushes = 0;
        for (tuple<int, int> edge:edges_batch) {
            edge_uu = get<0>(edge);
            edge_vv = get<1>(edge);
            worker->pre_deg[edge_uu]++;
            if (dlp->p[edge_uu] == 0.0)
                continue;
            assert((worker->pre_deg[edge_uu] - 1.0) > 0.0);
            delta = dlp->p[edge_uu] / (worker->pre_deg[edge_uu] - 1.0);
            dlp->p[edge_uu] += delta;
            dlp->r[edge_uu] -= delta / alpha;
            dlp->r[edge_vv] += (delta / alpha - delta);
        }
        front_posi = 0, rear_posi = 0;
        memset(&worker->mark_posi, 0, mark_size);
        for (uu = 0; uu < MAX_NODE_ID; uu++) {
            if (dlp->r[uu] <= eps_vec[uu])
                continue;
            worker->q_posi[rear_posi++] = uu;
            worker->mark_posi[uu] = true;
        }
        while (front_posi != rear_posi) { // positive queue
            num_pushes++;
            uu = worker->q_posi[front_posi++ % MAX_NODE_ID];
            res_uu = dlp->r[uu];
            dlp->p[uu] += alpha * res_uu;
            rest_prob = (1.0 - alpha) / out_deg[uu];
            for (auto v:graph[uu]) {
                dlp->r[v] += rest_prob * res_uu;
                if ((dlp->r[v] > eps_vec[v]) && !worker->mark_posi[v]) {
                    worker->q_posi[rear_posi++ % MAX_NODE_ID] = v;
                    worker->mark_posi[v] = true;
                }
            }
            dlp->r[uu] = 0.0;
            worker->mark_posi[uu] = false;
        }
        front_nega = 0, rear_nega = 0;
        memset(&worker->mark_nega, 0, mark_size);
        for (uu = 0; uu < MAX_NODE_ID; uu++) {
            if (dlp->r[uu] >= -eps_vec[uu])
                continue;
            worker->q_nega[rear_nega++] = uu;
            worker->mark_nega[uu] = true;
        }
        while (front_nega != rear_nega) { // negative queue
            num_pushes++;
            uu = worker->q_nega[front_nega++ % MAX_NODE_ID];
            res_uu = dlp->r[uu];
            dlp->p[uu] += alpha * res_uu;
            rest_prob = (1.0 - alpha) / out_deg[uu];
            for (auto v:graph[uu]) {
                dlp->r[v] += rest_prob * res_uu;
                if ((dlp->r[v] < -eps_vec[v]) && !worker->mark_nega[v]) {
                    worker->q_nega[rear_nega++ % MAX_NODE_ID] = v;
                    worker->mark_nega[v] = true;
                }
            }
            dlp->r[uu] = 0.0;
            worker->mark_nega[uu] = false;
        }
        calculate_emb(s);
        if (verbose <= 0)
            continue;
        cout << fixed << setprecision(6) << scientific;
        cout << "snapshot-" << snapshot_index << " thread-" << worker->th_id
             << " s: " << dlp->s << " cpu-time: " << cpu_time(cpu_t_start)
             << " sec " << " wall-time: " << wall_time(wall_t_start) << " sec "
             << " est: " << dlp->get_est() << " res: " << dlp->get_res()
             << " eps: " << eps_prime << " num-pushes: " << num_pushes
             << endl;
    }
    return nullptr;
}

static inline void *thread_local_forward_push_init(void *th_para) {
    // initialize a PPR.
    int *worker_id = (int *) (th_para);
    ppr_worker *worker = workers[*worker_id];
    int uu, front, rear, num_pushes;
    d_type res_uu, rest_prob;
    clock_t cpu_t_start = clock();
    auto wall_t_start = chrono::steady_clock::now();
    for (int s:worker->target_nodes) {
        if (estimators.find(s) != estimators.end() || out_deg[s] <= 0)
            continue;
        if (verbose == 1)
            cout << "start to push: " << s << endl;
        auto *dlp = new dyn_lp(s);
        front = 0, rear = 0, num_pushes = 0;
        memset(&worker->mark_posi, 0, sizeof(bool) * MAX_NODE_ID);
        worker->q_posi[rear++] = s;
        dlp->r[s] = 1.0;
        worker->mark_posi[s] = true;
        while (front != rear) {
            num_pushes++;
            uu = worker->q_posi[front++ % MAX_NODE_ID];
            res_uu = dlp->r[uu];
            dlp->p[uu] += alpha * res_uu;
            rest_prob = (1.0 - alpha) / out_deg[uu];
            for (auto v:graph[uu]) {
                dlp->r[v] += rest_prob * res_uu;
                if ((dlp->r[v] > eps_vec[v]) && !worker->mark_posi[v]) {
                    worker->q_posi[rear++ % MAX_NODE_ID] = v;
                    worker->mark_posi[v] = true;
                }
            }
            dlp->r[uu] = 0.0;
            worker->mark_posi[uu] = false;
        }
        estimators[s] = dlp;
        calculate_emb(s);
        if (verbose <= 0)
            continue;
        cout << fixed << setprecision(6) << scientific;
        cout << "snapshot-" << snapshot_index << " thread-" << worker->th_id
             << " s: " << dlp->s << " cpu-time: " << cpu_time(cpu_t_start)
             << " sec " << " wall-time: " << wall_time(wall_t_start) << " sec "
             << " est: " << dlp->get_est() << " res: " << dlp->get_res()
             << " eps: " << eps_prime << " num-pushes: " << num_pushes << endl;
    }
    return nullptr;
}

void distribute_works() {
    // active nodes will be uniformly distributed to workers
    uint64_t act_size = work_nodes.size();
    if (act_size <= num_cpus) {
        for (uint32_t ii = 0; ii < act_size; ii++) {
            workers[ii]->target_nodes.clear();
            workers[ii]->target_nodes.push_back(work_nodes[ii]);
        }
    } else {
        uint32_t step = act_size / num_cpus;
        uint32_t num_residual = act_size % num_cpus;
        uint32_t start, end;
        // make sure number of tasks in each worker uniformly distributed.
        for (uint32_t ii = 0; ii < num_cpus; ii++) {
            if (ii < num_residual) {
                start = ii * (step + 1);
                end = (ii + 1) * (step + 1);
            } else {
                start = (ii) * (step) + num_residual;
                end = (ii + 1) * (step) + num_residual;
            }
            workers[ii]->target_nodes.clear();
            for (uint32_t jj = start; jj < end; jj++)
                workers[ii]->target_nodes.push_back(work_nodes[jj]);
        }
    }
}

void parallel_initial() {
    clock_t cpu_t_start = clock();
    auto wall_t_start = chrono::steady_clock::now();
    void *(*func)(void *);
    func = thread_local_forward_push_init;
    work_nodes.clear();
    for (int s:emb_nodes) {
        if (estimators.find(s) != estimators.end() || out_deg[s] <= 0)
            continue;
        work_nodes.push_back(s);
        if (verbose == 1)
            cout << "initialize node: " << s
                 << " deg[s]: " << out_deg[s] << endl;
    }
    distribute_works();
    pthread_t th[num_cpus];
    int th_ids[num_cpus];
    for (uint32_t i = 0; i < num_cpus; i++) {
        th_ids[i] = i;
        pthread_create(&th[i], nullptr, func, (void *) (th_ids + i));
    }
    for (uint32_t th_id = 0; th_id < num_cpus; th_id++)
        pthread_join(th[th_id], nullptr);
    run_time_push_cpu_time += cpu_time(cpu_t_start);
    run_time_push_wall_time += wall_time(wall_t_start);
}

void parallel_push_forward() {
    clock_t cpu_t_start = clock();
    auto wall_t_start = chrono::steady_clock::now();
    void *(*func)(void *);
    func = thread_local_forward_push_batch;
    work_nodes.clear();
    for (int s:emb_nodes) {
        if (estimators.find(s) == estimators.end()) continue;
        work_nodes.push_back(s);
    }
    distribute_works();
    pthread_t th[num_cpus];
    int th_ids[num_cpus];
    for (uint32_t i = 0; i < num_cpus; i++) {
        th_ids[i] = i;
        pthread_create(&th[i], nullptr, func, (void *) (th_ids + i));
    }
    for (uint32_t th_id = 0; th_id < num_cpus; th_id++)
        pthread_join(th[th_id], nullptr);
    run_time_push_cpu_time += cpu_time(cpu_t_start);
    run_time_push_wall_time += wall_time(wall_t_start);
}


void read_target_nodes(const string &target_nodes_path) {
    int vv, node_index = 0;
    emb_nodes.clear();
    ifstream f_read;
    f_read.open(target_nodes_path.c_str());
    while (f_read >> vv) {
        assert(0 <= vv && vv <= MAX_NODE_ID);
        emb_nodes.push_back(vv);
        emb_nodes_indices[vv] = node_index;
        node_index++;
    }
    f_read.close();
}

void save_statistics() {
    ofstream file(re_stat_path, fstream::app);
    stringstream out_str;
    out_str << fixed << setprecision(6) << scientific;
    out_str << "snapshot " << snapshot_index;
    out_str << " #edges " << num_edges << " cpu " << num_cpus;
    out_str << " edge-time " << run_time_io << " ";
    out_str << " eps-prime " << eps_prime << " ";
    out_str << " push-cpu-time " << run_time_push_cpu_time << " ";
    out_str << " push-wall-time " << run_time_push_wall_time << " ";
    out_str << " t-cpu-time " << run_time_total_cpu_time;
    out_str << " t-wall-time " << run_time_total_wall_time << endl;
    file << out_str.str();
    cout << out_str.str();
    file.close();
    fflush(stdout);
}

void create_config_paths() {
    char root_config[MAX_STR_LEN];
    strcpy(root_config, config_path);
    ifstream config_file(strcat(root_config, "/config.txt"));
    // target-node path, g0 path, g1 path, ...
    copy(istream_iterator<string>(config_file),
         istream_iterator<string>(), back_inserter(list_g_paths));
    assert(list_g_paths.size() >= 2); // at least has g0
    num_snapshots = list_g_paths.size() - 1;
    assert(num_snapshots > 0);
    stringstream f_re_emb, f_re_stat;
    f_re_emb.precision(2);
    f_re_emb << scientific << config_path << "/emb_dynamic-ppe_d:"
             << emb_dim << "_eps:" << epsilon << "_alpha:" << alpha << "_t:"
             << num_snapshots << "_seed:" << seed << ".bin";
    strcpy(re_emb_path, f_re_emb.str().c_str());
    f_re_stat.precision(2);
    f_re_stat << scientific << config_path << "/stat_dynamic-ppe_d:"
              << emb_dim << "_eps:" << scientific << epsilon << "_alpha:"
              << alpha << "_t:" << num_snapshots << "_seed:" << seed << ".txt";
    strcpy(re_stat_path, f_re_stat.str().c_str());
    ofstream file;
    file.open(re_stat_path, ofstream::out | ofstream::trunc);
    file.close();
}

void create_workers() {
    for (uint32_t ii = 0; ii < num_cpus; ii++) {
        auto *worker = new ppr_worker(ii);
        worker->target_nodes.clear();
        workers[ii] = worker;
    }
}

void initialize_projection() {
    // use two fixed hash or randomly generate from MurMurHash3 family
    // https://github.com/aappleby/smhasher
    if (strcmp(model, "hash") == 0)
        if (seed < 0)
            for (uint32_t ii = 0; ii < MAX_NODE_ID; ii++) {
                h1_arr[ii] = rj_hash(ii) % emb_dim;
                h2_arr[ii] = rj_hash(ii) % 2;
            }
        else
            for (uint32_t ii = 0; ii < MAX_NODE_ID; ii++) {
                MurmurHash3_x86_32(&ii, sizeof(uint32_t), seed, &h1_arr[ii]);
                MurmurHash3_x86_32(&ii, sizeof(uint32_t), seed, &h2_arr[ii]);
                h1_arr[ii] = h1_arr[ii] % emb_dim;
                h2_arr[ii] = h2_arr[ii] % 2;
            }
    else {
        rand_proj_size = MAX_NODE_ID * emb_dim;
        d_type mean = 0.0, std = 1. / sqrt(emb_dim);
        rand_proj = (d_type *) malloc(rand_proj_size * sizeof(d_type));
        for (uint64_t ii = 0; ii < rand_proj_size; ii++)
            emb[ii] = gaussian_sample(mean, std);
    }
}

void initialize_graph() {
    num_edges = 0;
    nodes.clear();
    for (int ii = 0; ii < MAX_NODE_ID; ii++)
        graph.emplace_back();
    for (int ii = 0; ii < MAX_NODE_ID; ii++)
        graph[ii].clear();
    fill(begin(out_deg), end(out_deg), 0);
}

void initialize_emb() {
    snapshot_size = emb_nodes.size() * emb_dim;
    emb_size = snapshot_size * (num_snapshots);
    emb = (d_type *) calloc(emb_size, sizeof(d_type));
    d_type mean = 0.0, std = 1. / sqrt(emb_dim);
    for (uint32_t ii = 0; ii < emb_size; ii++)
        emb[ii] = gaussian_sample(mean, std);
}

void print_algo_config() {
    cout << "-------------- dynamic-ppe configuration ---------------" << endl;
    cout << "alpha: " << alpha << endl;
    cout << "epsilon: " << epsilon << endl;
    cout << "random seed: " << seed << endl;
    cout << "number of cpus: " << num_cpus << endl;
    cout << "verbose level: " << verbose << endl;
    cout << "projection method: " << model << endl;
    cout << "embedding dimension: " << emb_dim << endl;
    cout << "num-target-nodes: " << emb_nodes.size() << endl;
    cout << "snapshot_size: " << snapshot_size << endl;
    cout << "number of snapshots: " << num_snapshots << endl;
    cout << "embedding size: " << emb_size << endl;
    cout << "maximum node id allowed: " << MAX_NODE_ID << endl;
    cout << "config path: " << config_path << endl;
    cout << "embedding save to: " << re_emb_path << endl;
    cout << "statistic save to: " << re_stat_path << endl;
    cout << "--------------------------------------------------------" << endl;
}

int main(int argc, char **argv) {
    clock_t cpu_t_start = clock();
    auto wall_t_start = chrono::steady_clock::now();
    run_time_io = 0.0; // io time for reading edges
    run_time_push_cpu_time = 0.0;
    run_time_push_wall_time = 0.0;
    run_time_total_cpu_time = 0.0;
    run_time_total_wall_time = 0.0;
    fstream f_stream;
    if (argc <= 8) {
        cout << "DynamicPPE: ./dynamic-ppe-fast config_path emb_dim "
                "epsilon alpha model seed num_cpus verbose" << endl;
        exit(-1);
    }
    strcpy(config_path, argv[1]);
    assert(strlen(config_path) > 0);
    emb_dim = stoi(argv[2]);
    assert(emb_dim >= 2 && "Embedding dim should >=2.");
    epsilon = stod(argv[3]);
    assert(0.0 < epsilon);
    alpha = stod(argv[4]);
    assert(0.0 < alpha && alpha < 1.0);
    model = argv[5];
    assert(strcmp(model, "hash") == 0 || strcmp(model, "rand") == 0);
    seed = stoi(argv[6]);
    num_cpus = stoi(argv[7]);
    assert(num_cpus > 0);
    verbose = stoi(argv[8]);
    assert(verbose == 0 || verbose == 1);
    create_config_paths();
    read_target_nodes(list_g_paths[0]);
    generator = new default_random_engine(seed);
    uni_dist = new uniform_real_distribution<>();
    create_workers();
    initialize_projection();
    initialize_graph();
    initialize_emb();
    print_algo_config();
    snapshot_index = 0;
    batch_update_graph(list_g_paths[1]);
    if (!emb_nodes.empty())
        parallel_initial();
    run_time_total_cpu_time = cpu_time(cpu_t_start);
    run_time_total_wall_time = wall_time(wall_t_start);
    save_statistics();
    snapshot_index++;
    // --- dynamic update graph from t=1, 2, ..., to T
    for (size_t i = 1; i < num_snapshots; i++) {
        batch_update_graph(list_g_paths[1 + i]);
        parallel_push_forward();
        if (estimators.size() != emb_nodes.size())
            parallel_initial();
        run_time_total_cpu_time = cpu_time(cpu_t_start);
        run_time_total_wall_time = wall_time(wall_t_start);
        save_statistics();
        snapshot_index++;
    }
    cout << "Write embedding to " << re_emb_path << endl;
    FILE *f = fopen(re_emb_path, "wb");
    fwrite(emb, sizeof(d_type), emb_size, f);
    fclose(f);
    free(emb);
    if (rand_proj != nullptr)
        free(rand_proj);
    for (auto node: emb_nodes)
        if (estimators.find(node) != estimators.end())
            delete estimators[node];
    for (auto worker:workers)
        delete workers[worker.first];
    delete generator;
    delete uni_dist;
    return 0;
}