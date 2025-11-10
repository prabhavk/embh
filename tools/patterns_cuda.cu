// patterns_cuda.cu
// Build unique site-patterns from a FASTA alignment using CUDA to
// hash each column and count A/C/G/T in parallel, after replacing
// all non-ACGT characters with a gap '-'.
//
// Outputs:
//   <prefix>.pat       (each line: "<weight> <pattern>", pattern encoded as integers 0-4)
//   <prefix>.basecomp  (weighted A/C/G/T composition, plus gap stats)
//
// Compile: nvcc -O3 -std=c++11 patterns_cuda.cu -o patterns_cuda
// Usage:   ./patterns_cuda input.fasta [output_prefix]

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
    exit(1); \
  } \
} while (0)

static void die(const char* msg) {
  fprintf(stderr, "Error: %s\n", msg);
  exit(1);
}

/* ---------------- FASTA reading ---------------- */

struct Seq { std::string name, seq; };
struct SeqVec { std::vector<Seq> v; };

static void read_fasta(const char* path, SeqVec& out) {
  FILE* f = fopen(path, "r");
  if (!f) die("cannot open FASTA");

  char* line = nullptr; size_t cap=0; ssize_t len;
  std::string cur_name, cur_seq;

  while ((len = getline(&line, &cap, f)) != -1) {
    if (len>0 && (line[len-1]=='\n' || line[len-1]=='\r')) line[--len]='\0';
    if (len>0 && line[0] == '>') {
      if (!cur_name.empty()) {
        out.v.push_back({cur_name, cur_seq});
        cur_name.clear(); cur_seq.clear();
      }
      cur_name.assign(line+1);
    } else if (len>0) {
      if (cur_name.empty()) die("sequence before header");
      for (ssize_t i=0;i<len;i++) {
        char c = line[i];
        if (!std::isspace((unsigned char)c)) cur_seq.push_back(c);
      }
    }
  }
  if (!cur_name.empty()) out.v.push_back({cur_name, cur_seq});
  free(line); fclose(f);
  if (out.v.empty()) die("empty FASTA");
  // Validate equal lengths
  size_t L = out.v[0].seq.size();
  for (size_t i=1;i<out.v.size();++i) {
    if (out.v[i].seq.size() != L) die("all sequences must have same length");
  }
}

/* ---------------- GPU kernel: per-column hash + A/C/G/T counts + gap stats ---------------- */

__device__ __forceinline__
char to_upper(char c) {
  if (c >= 'a' && c <= 'z') return c - 32;
  return c;
}

// 64-bit FNV-1a constants
__device__ __constant__ unsigned long long FNV_OFFSET = 1469598103934665603ULL;
__device__ __constant__ unsigned long long FNV_PRIME  = 1099511628211ULL;

__global__ void hash_columns_kernel(
    const char* __restrict__ seqmat,   // [nseq x L], row-major by sequence
    size_t nseq, size_t L,
    unsigned long long* __restrict__ hashes, // [L]
    int* __restrict__ A, int* __restrict__ C, int* __restrict__ G, int* __restrict__ T, // [L]
    int* __restrict__ gaps_orig,            // [L] count of original gaps ('-' or '.')
    int* __restrict__ non_acgt_converted)   // [L] count of non-ACGT, non-gap converted to '-'
{
  size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= L) return;

  unsigned long long h = FNV_OFFSET;
  int a=0,c=0,g=0,t=0;
  int g_orig=0, conv=0;

  for (size_t s=0; s<nseq; ++s) {
    char raw = seqmat[s*L + pos];
    char up  = to_upper(raw);

    bool is_gap_orig = (raw == '-' || raw == '.');
    bool is_acgt = (up=='A' || up=='C' || up=='G' || up=='T');

    char used; // character used for hashing/pattern after conversion
    if (is_acgt) {
      used = up;
      if      (up=='A') a++;
      else if (up=='C') c++;
      else if (up=='G') g++;
      else              t++; // 'T'
    } else {
      // treat everything non-ACGT as gap '-'
      used = '-';
      if (is_gap_orig) g_orig++;
      else             conv++;
    }

    // include transformed char 'used' in the hash
    h ^= (unsigned long long)(unsigned char)used;
    h *= FNV_PRIME;
  }

  hashes[pos] = h;
  A[pos] = a; C[pos] = c; G[pos] = g; T[pos] = t;
  gaps_orig[pos] = g_orig;
  non_acgt_converted[pos] = conv;
}

/* ---------------- Host aggregation with collision check ---------------- */

struct PatAgg {
  std::vector<int> key; // transformed pattern as integers (0=A, 1=C, 2=G, 3=T, 4=gap)
  int weight = 0;
  long long a=0,c=0,g=0,t=0; // aggregated base counts (per pattern × weight)
};

// Convert character to integer encoding: A=0, C=1, G=2, T=3, gap=4
static int char_to_int(char c) {
  char up = (char)std::toupper((unsigned char)c);
  if (up == 'A') return 0;
  if (up == 'C') return 1;
  if (up == 'G') return 2;
  if (up == 'T') return 3;
  return 4; // gap or non-ACGT
}

static std::vector<int> column_string_transformed(const SeqVec& sv, size_t pos) {
  std::vector<int> col;
  col.resize(sv.v.size());
  for (size_t s=0; s<sv.v.size(); ++s) {
    char raw = sv.v[s].seq[pos];
    col[s] = char_to_int(raw);
  }
  return col;
}

static void make_output_prefix(const char* in, const char* opt, char* out, size_t outsz){
  if (opt && *opt) { snprintf(out, outsz, "%s", opt); return; }
  const char* base = strrchr(in, '/'); base = base? base+1 : in;
  char tmp[1024]; snprintf(tmp, sizeof(tmp), "%s", base);
  char* dot = strrchr(tmp, '.'); if (dot) *dot = '\0';
  snprintf(out, outsz, "%s", tmp);
}

/* ----------------------------- main ----------------------------- */

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <input.fasta> [output_prefix]\n", argv[0]);
    return 1;
  }
  const char* fasta_path = argv[1];
  const char* out_prefix_arg = (argc>=3? argv[2] : nullptr);

  // Read FASTA
  SeqVec sv; read_fasta(fasta_path, sv);
  const size_t nseq = sv.v.size();
  const size_t L    = sv.v[0].seq.size();

  // Pack into contiguous matrix [nseq x L]
  std::vector<char> hostMat(nseq * L);
  for (size_t s=0; s<nseq; ++s) {
    memcpy(&hostMat[s*L], sv.v[s].seq.data(), L);
  }

  // Allocate device buffers
  char* d_seqmat = nullptr;
  unsigned long long* d_hashes = nullptr;
  int *dA=nullptr, *dC=nullptr, *dG=nullptr, *dT=nullptr;
  int *dGapsOrig=nullptr, *dConverted=nullptr;

  CUDA_CHECK(cudaMalloc(&d_seqmat, nseq * L * sizeof(char)));
  CUDA_CHECK(cudaMalloc(&d_hashes, L * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&dA, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dC, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dG, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dT, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dGapsOrig, L * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dConverted, L * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_seqmat, hostMat.data(), nseq * L * sizeof(char), cudaMemcpyHostToDevice));

  // Launch kernel: 1 thread per column
  const int BLOCK = 256;
  const int GRID  = (int)((L + BLOCK - 1) / BLOCK);
  hash_columns_kernel<<<GRID, BLOCK>>>(d_seqmat, nseq, L, d_hashes, dA, dC, dG, dT, dGapsOrig, dConverted);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back results
  std::vector<unsigned long long> hashes(L);
  std::vector<int> A(L), C(L), G(L), T(L);
  std::vector<int> GapsOrig(L), Converted(L);
  CUDA_CHECK(cudaMemcpy(hashes.data(),    d_hashes,    L*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(A.data(),         dA,          L*sizeof(int),               cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(C.data(),         dC,          L*sizeof(int),               cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(G.data(),         dG,          L*sizeof(int),               cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(T.data(),         dT,          L*sizeof(int),               cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(GapsOrig.data(),  dGapsOrig,   L*sizeof(int),               cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Converted.data(), dConverted,  L*sizeof(int),               cudaMemcpyDeviceToHost));

  cudaFree(d_seqmat); cudaFree(d_hashes);
  cudaFree(dA); cudaFree(dC); cudaFree(dG); cudaFree(dT);
  cudaFree(dGapsOrig); cudaFree(dConverted);

  // Totals for basecomp report
  long long total_gaps_original = 0;
  long long total_non_acgt_converted = 0;
  for (size_t i=0;i<L;i++) {
    total_gaps_original      += GapsOrig[i];
    total_non_acgt_converted += Converted[i];
  }

  // Group by hash; verify equality using transformed column string
  std::unordered_map<unsigned long long, std::vector<size_t>> groups;
  groups.reserve(L*1.3);
  for (size_t pos=0; pos<L; ++pos) groups[hashes[pos]].push_back(pos);

  // Hash function for vector<int>
  struct VectorIntHash {
    std::size_t operator()(const std::vector<int>& v) const {
      std::size_t seed = v.size();
      for (auto& i : v) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };

  std::unordered_map<std::vector<int>, PatAgg, VectorIntHash> pats;
  pats.reserve(groups.size()*2);

  for (auto& kv : groups) {
    auto& idxs = kv.second;
    for (size_t pos : idxs) {
      std::vector<int> key = column_string_transformed(sv, pos);
      auto it = pats.find(key);
      if (it == pats.end()) {
        PatAgg agg;
        agg.key    = key;
        agg.weight = 1;
        agg.a = A[pos]; agg.c = C[pos]; agg.g = G[pos]; agg.t = T[pos];
        pats.emplace(agg.key, std::move(agg));
      } else {
        it->second.weight += 1;
        it->second.a += A[pos];
        it->second.c += C[pos];
        it->second.g += G[pos];
        it->second.t += T[pos];
      }
    }
  }

  // Sort patterns by decreasing weight
  std::vector<PatAgg> out;
  out.reserve(pats.size());
  for (auto& kv : pats) out.push_back(std::move(kv.second));
  std::sort(out.begin(), out.end(), [](const PatAgg& x, const PatAgg& y){
    if (x.weight != y.weight) return x.weight > y.weight;
    return x.key < y.key;
  });

  // Output files
  char prefix[1024];
  auto make_prefix = [&](const char* in, const char* opt){
    if (opt && *opt) { snprintf(prefix, sizeof(prefix), "%s", opt); return; }
    const char* base = strrchr(in, '/'); base = base? base+1 : in;
    char tmp[1024]; snprintf(tmp, sizeof(tmp), "%s", base);
    char* dot = strrchr(tmp, '.'); if (dot) *dot = '\0';
    snprintf(prefix, sizeof(prefix), "%s", tmp);
  };
  make_prefix(fasta_path, out_prefix_arg);

  char path_pat[1200], path_bc[1200];
  snprintf(path_pat, sizeof(path_pat), "%s.pat",      prefix);     // "<weight> <pattern>"
  snprintf(path_bc,  sizeof(path_bc),  "%s.basecomp", prefix);

  FILE* fpat = fopen(path_pat, "w");   if (!fpat) die("cannot write .pat");
  FILE* fbc  = fopen(path_bc,  "w");   if (!fbc)  die("cannot write .basecomp");

  // Patterns file: "<weight> <pattern>" with integers
  for (const auto& p : out) {
    fprintf(fpat, "%d", p.weight);
    for (size_t i = 0; i < p.key.size(); ++i) {
      fprintf(fpat, " %d", p.key[i]);
    }
    fprintf(fpat, "\n");
  }

  // Weighted base composition (after conversion; gaps ignored)
  long long TA=0, TC=0, TG=0, TT=0;
  for (const auto& p : out) {
    TA += p.a; TC += p.c; TG += p.g; TT += p.t;
  }
  long long TOT = TA + TC + TG + TT;
  double fA=0, fC=0, fG=0, fT=0;
  if (TOT > 0) {
    fA = (double)TA / (double)TOT;
    fC = (double)TC / (double)TOT;
    fG = (double)TG / (double)TOT;
    fT = (double)TT / (double)TOT;
  }
  
  // Updated format: use integer encoding in comments
  fprintf(fbc, "0\t%.10f\t(%lld)\n", fA, TA);  // A=0
  fprintf(fbc, "1\t%.10f\t(%lld)\n", fC, TC);  // C=1
  fprintf(fbc, "2\t%.10f\t(%lld)\n", fG, TG);  // G=2
  fprintf(fbc, "3\t%.10f\t(%lld)\n", fT, TT);  // T=3
  fprintf(fbc, "TOTAL_ACGT\t%lld\n", TOT);
  fprintf(fbc, "NSEQ\t%zu\n", nseq);
  fprintf(fbc, "NSITES\t%zu\n", L);
  fprintf(fbc, "GAPS_ORIGINAL\t%lld\n", total_gaps_original);
  fprintf(fbc, "NON_ACGT_CONVERTED_TO_GAP\t%lld\n", total_non_acgt_converted);

  fclose(fpat); fclose(fbc);

  fprintf(stderr, "Wrote:\n  %s  (weight + pattern)\n  %s  (base comp + gap stats)\n",
          path_pat, path_bc);
  return 0;
}
