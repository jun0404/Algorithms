//Template
#include <bits/stdc++.h>
using namespace std;
#define all(x) begin(x), end(x)
#define MOD 1000000007
#define EPS 1e-3
#define rep(i, a, b) for(int i = a; i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef complex<double> C;
typedef vector<double> vd;
typedef long long ll;

int main() {
	ios_base::sync_with_stdio();
	cin.tie();

	return 0;
}

====================
//Method of Solution
/*
1. Greedy = O(1) or Sort
 - O(1) = Mathematical Regularity/Dividing Cases/Brute Force
2. DP uses positions, states(+Bitmask), or values itself(+compression+GCD/LCM)
*/


//RULES
/*
1. Do the mathematical proof first.
2. Do not hesitate to go back to the start. Don't try to find what's wrong.
3. Print all the variables for debugging
4. Corner Case - 0-1 Index, End-to-End
5. Stress Test
*/

//Time complexity
/*
max value of n   time complexity
   10^6              O(n)
   10^5              O(nlogn)
   10^3              O(n^2)
   10^2              O(n^3)
   10^9              O(log n)
*/

/*
Pre-submit:
	Write a few simple test cases if sample is not enough.
	Are time limits close? If so, generate max cases.
	Is the memory usage fine?
	Could anything overflow?
	Make sure to submit the right file.
Wrong answer:
	Print your solution! Print debug output, as well.
	Are you clearing all data structures between test cases?
	Can your algorithm handle the whole range of input?
	Read the full problem statement again.
	Do you handle all corner cases correctly?
	Have you understood the problem correctly?
	Any uninitialized variables?
	Any overflows?
	Confusing N and M, i and j, etc.?
	Are you sure your algorithm works?
	What special cases have you not thought of?
	Are you sure the STL functions you use work as you think?
	Add some assertions, maybe resubmit.
	Create some testcases to run your algorithm on.
	Go through the algorithm for a simple case.
	Go through this list again.
	Explain your algorithm to a teammate.
	Ask the teammate to look at your code.
	Go for a small walk, e.g. to the toilet.
	Is your output format correct? (including whitespace)
	Rewrite your solution from the start or let a teammate do it.
Runtime error:
	Have you tested all corner cases locally?
	Any uninitialized variables?
	Are you reading or writing outside the range of any vector?
	Any assertions that might fail?
	Any possible division by 0? (mod 0 for example)
	Any possible infinite recursion?
	Invalidated pointers or iterators?
	Are you using too much memory?
	Debug with resubmits (e.g. remapped signals, see Various).
	Time limit exceeded:
	Do you have any possible infinite loops?
	What is the complexity of your algorithm?
	Are you copying a lot of unnecessary data? (References)
	How big is the input and output? (consider scanf)
	Avoid vector, map. (use arrays/unordered_map)
	What do your teammates think about your algorithm?
Memory limit exceeded:
	What is the max amount of memory your algorithm should need?
	Are you clearing all data structures between test cases?
*/

================
//Binary Exponentiation
long long binpow(long long a, long long b) { //a^b
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a;
        a = a * a;
        b >>= 1;
    }
    return res;
}

================
//Binary Search

bool search(int x[], int n, int k) {
    int p = 0;
    for (int a = n; a >= 1; a /= 2) {
        while (p+a < n && x[p+a] <= k) p += a;
    }
    return x[p] == k;
}

================

//Combination
#include <algorithm>
#include <iostream>
#include <string>

void comb(int N, int K)
{
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's

    // print integers and permute bitmask
    do {
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i]) std::cout << " " << i;
        }
        std::cout << std::endl;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

int main()
{
    comb(5, 3);
}

============
//Modular Arithmetic

long long int gcd (long long int a, long long int b) {
    return b ? gcd (b, a % b) : a;
}

long long int ext_euclid(long long int a, long long int b, long long int& x, long long int& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    long long int x1, y1;
    long long int d = ext_euclid(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}

long long int factmod(long long int n, long long int p) {
    vector<long long int> f(p);
    f[0] = 1;
    for (long long int i = 1; i < p; i++)
        f[i] = f[i-1] * i % p;

    long long int res = 1;
    while (n > 1) {
        if ((n/p) % 2)
            res = p - res;
        res = res * f[n%p] % p;
        n /= p;
    }
    return res; 
}

long long int multiplicity_factorial(long long int n, long long int p) {
    long long int count = 0;
    do {
        n /= p;
        count += n;
    } while (n);
    return count;
}

//Modpow
const ll mod = 1000000007; // faster if const

ll modpow(ll b, ll e) {
	ll ans = 1;
	for (; e; b = b * b % mod, e /= 2)
		if (e & 1) ans = ans * b % mod;
	return ans;
}

//ModInv can be done with binary exponentiation by O(logm) or with the below by O(m)
// const ll mod = 1000000007, LIM = 200000; ///include-line
ll* inv = new ll[LIM] - 1; inv[1] = 1;
rep(i,2,LIM) inv[i] = mod - (mod / i) * inv[mod % i] % mod;

//Binomial Coefficient Modular

factorial[0] = 1;
for (int i = 1; i <= MAXN; i++) {
    factorial[i] = factorial[i - 1] * i % m;
}

long long binomial_coefficient(int n, int k) {
    return factorial[n] * inverse(factorial[k] * factorial[n - k] % m) % m;
}

long long binomial_coefficient(int n, int k) {
    return factorial[n] * inverse_factorial[k] % m * inverse_factorial[n - k] % m;
}

//Euler Totient Function

long long int phi(long long int n) {
    long long int result = n;
    for (long long int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}
==============
//XOR Linear Algebra

//f is the position of the first 1 occurring in binary representation

int basis[d]; // basis[i] keeps the mask of the vector whose f value is i

int sz; // Current size of the basis

void insertVector(int mask) {
	for (int i = 0; i < d; i++) {
		if ((mask & 1 << i) == 0) continue; // continue if i != f(mask)

		if (!basis[i]) { // If there is no basis vector with the i'th bit set, then insert this vector into the basis
			basis[i] = mask;
			++sz;
			
			return;
		}

		mask ^= basis[i]; // Otherwise subtract the basis vector from this vector
	}
}

==================================================================================

//One-based Segment Tree

// tree size
int tree[1 << 18], lazy[1 << 18];

/*
n: position client wants to update
v: updating value
bit: current position
s: starting point of the current position
e: ending point of the current position
*/

void make (int n, int v, int bit = 1, int s = 1, int e = 10000) {
	int m = (s + e) >> 1;

	// are you at the leaf?
	if (s == e) {
		tree[bit] = v;
		return;
	}
	if (n <= m) {
		make (n, v, 2 * bit, s, m);
	} else {
		make (n, v, 2 * bit + 1, m + 1, e);
	}
	// update from the leaf!
	tree[bit] = tree[2 * bit] + tree[2 * bit + 1];
	return;
}

int get_sum (int n1, int n2, int bit = 1, int s = 1, int e = 10000) {
	int m = (s + e) >> 1;
	if (n2 < n1 || n2 < s || e < n1) {
		return 0;
	}
	if (n1 <= s && e <= n2) {
		return tree[bit];
	}
	return get_sum(n1, n2, 2 * bit, s, m) + get_sum (n1, n2, 2 * bit + 1, m + 1, e);
}

== //lazy

void lazy_propagation(int bit, int s, int e)
{
   tree[bit] += lazy[bit] * (e - s + 1);
   if (s < e) {
      lazy[2 * bit] += lazy[bit];
      lazy[2 * bit + 1] += lazy[bit];
   }
   lazy[bit] = 0;
}

void add_tree(int n1, int n2, int v, int bit = 1, int s = 1, int e = 100000)
{
   int m = (s + e) >> 1;
   lazy_propagation(bit, s, e);
   if (n2 < n1 || n2 < s || e < n1) {
      return;
   }
   if (n1 <= s && e <= n2) {
      tree[bit] += v * (e - s + 1);
      if (s < e) {
         lazy[2 * bit] += v;
         lazy[2 * bit + 1] += v;
      }
      return;
   }
   add_tree(n1, n2, v, 2 * bit, s, m);
   add_tree(n1, n2, v, 2 * bit + 1, m + 1, e);
   tree[bit] = tree[2 * bit] + tree[2 * bit + 1];
}

int get_sum(int n1, int n2, int bit = 1, int s = 1, int e = 100000)
{
   int m = (s + e) >> 1;
   lazy_propagation(bit, s, e);
   if (n2 < n1 || n2 < s || e < n1) {
      return 0;
   }
   if (n1 <= s && e <= n2) {
      return tree[bit];
   }
   return get_sum(n1, n2, 2 * bit, s, m) + get_sum(n1, n2, 2 * bit + 1, m + 1, e);
}
=========================
//Heavy-Light Decomposition
/**
 * Author: Benjamin Qi, Oleksandr Kulkov, chilli
 * Date: 2020-01-12
 * License: CC0
 * Source: https://codeforces.com/blog/entry/53170, https://github.com/bqi343/USACO/blob/master/Implementations/content/graphs%20(12)/Trees%20(10)/HLD%20(10.3).h
 * Description: Decomposes a tree into vertex disjoint heavy paths and light
 * edges such that the path from any leaf to the root contains at most log(n)
 * light edges. Code does additive modifications and max queries, but can
 * support commutative segtree modifications/queries on paths and subtrees.
 * Takes as input the full adjacency list. VALS\_EDGES being true means that
 * values are stored in the edges, as opposed to the nodes. All values
 * initialized to the segtree default. Root must be 0.
 * Time: O((\log N)^2)
 * Status: stress-tested against old HLD
 */
static char buf[450 << 20];
void* operator new(size_t s) {
	static size_t i = sizeof buf;
	assert(s < i);
	return (void*)&buf[i -= s];
}
void operator delete(void*) {}

const int inf = 1e9;
struct Node {
	Node *l = 0, *r = 0;
	int lo, hi, mset = inf, madd = 0, val = -inf;
	Node(int lo,int hi):lo(lo),hi(hi){} // Large interval of -inf
	Node(vi& v, int lo, int hi) : lo(lo), hi(hi) {
		if (lo + 1 < hi) {
			int mid = lo + (hi - lo)/2;
			l = new Node(v, lo, mid); r = new Node(v, mid, hi);
			val = max(l->val, r->val);
		}
		else val = v[lo];
	}
	int query(int L, int R) {
		if (R <= lo || hi <= L) return -inf;
		if (L <= lo && hi <= R) return val;
		push();
		return max(l->query(L, R), r->query(L, R));
	}
	void set(int L, int R, int x) {
		if (R <= lo || hi <= L) return;
		if (L <= lo && hi <= R) mset = val = x, madd = 0;
		else {
			push(), l->set(L, R, x), r->set(L, R, x);
			val = max(l->val, r->val);
		}
	}
	void add(int L, int R, int x) {
		if (R <= lo || hi <= L) return;
		if (L <= lo && hi <= R) {
			if (mset != inf) mset += x;
			else madd += x;
			val += x;
		}
		else {
			push(), l->add(L, R, x), r->add(L, R, x);
			val = max(l->val, r->val);
		}
	}
	void push() {
		if (!l) {
			int mid = lo + (hi - lo)/2;
			l = new Node(lo, mid); r = new Node(mid, hi);
		}
		if (mset != inf)
			l->set(lo,hi,mset), r->set(lo,hi,mset), mset = inf;
		else if (madd)
			l->add(lo,hi,madd), r->add(lo,hi,madd), madd = 0;
	}
};

template <bool VALS_EDGES> struct HLD {
	int N, tim = 0;
	vector<vi> adj;
	vi par, siz, depth, rt, pos;
	Node *tree;
	HLD(vector<vi> adj_)
		: N(sz(adj_)), adj(adj_), par(N, -1), siz(N, 1), depth(N),
		  rt(N),pos(N),tree(new Node(0, N)){ dfsSz(0); dfsHld(0); }
	void dfsSz(int v) {
		if (par[v] != -1) adj[v].erase(find(all(adj[v]), par[v]));
		for (int& u : adj[v]) {
			par[u] = v, depth[u] = depth[v] + 1;
			dfsSz(u);
			siz[v] += siz[u];
			if (siz[u] > siz[adj[v][0]]) swap(u, adj[v][0]);
		}
	}
	void dfsHld(int v) {
		pos[v] = tim++;
		for (int u : adj[v]) {
			rt[u] = (u == adj[v][0] ? rt[v] : u);
			dfsHld(u);
		}
	}
	template <class B> void process(int u, int v, B op) {
		for (; rt[u] != rt[v]; v = par[rt[v]]) {
			if (depth[rt[u]] > depth[rt[v]]) swap(u, v);
			op(pos[rt[v]], pos[v] + 1);
		}
		if (depth[u] > depth[v]) swap(u, v);
		op(pos[u] + VALS_EDGES, pos[v] + 1);
	}
	void modifyPath(int u, int v, int val) {
		process(u, v, [&](int l, int r) { tree->add(l, r, val); });
	}
	int queryPath(int u, int v) { // Modify depending on problem
		int res = -1e9;
		process(u, v, [&](int l, int r) {
				res = max(res, tree->query(l, r));
		});
		return res;
	}
	int querySubtree(int v) { // modifySubtree is similar
		return tree->query(pos[v] + VALS_EDGES, pos[v] + siz[v]);
	}
};

===========================================================

//Edmond-Karp Algorithm
#include <iostream>
#include <limits.h>
#include <string.h>
#include <queue>
using namespace std;

bool bfs(int rGraph[V][V], int s, int t, int parent[]) {
	bool visited[V];
	memset(visited, 0, sizeof(visited));
	queue<int> q;
	q.push(s);
	visited[s] = true;
	parent[s] = -1;
	while (!q.empty()) {
		int u = q.front();
		q.pop();
		for (int v = 0; v < V; v++) {
			if (visited[v] == false && rGraph[u][v] > 0) {
				q.push(v);
				parent[v] = u;
				visited[v] = true;
			}
		}
	}
	return (visited[t] == true);
}

int solve(int graph[V][V], int s, int t) {
	int u, v;
	int rGraph[V][V];
	for (u = 0; u < V; u++) {
		for (v = 0; v < V; v++) {
			rGraph[u][v] = graph[u][v];
		}
	}
	int parent[V];
	int max_flow = 0;
	while (bfs(rGraph, s, t, parent)) {
		int path_flow = INT_MAX;
		for (v = t; v != s; v = parent[v]) {
			u = parent[v];
			path_flow = min(path_flow, rGraph[u][v]);
		}
		for (v = t; v != s; v = parent[v]) {
			u = parent[v];
			rGraph[u][v] -= path_flow;
			rGraph[v][u] += path_flow;
		}
		max_flow += path_flow;
	}
	return max_flow;
}

int main() {
	int N, M;
	for (int i = 1; i <= N; i++) {
		graph[0][i] = 1;
	}
	for (int i = 1; i <= M; i++) {
		graph[N+i][N+M+1] = 1;
	}
	for (int i = 1; i <= N; i++) {
		int how_many; scanf("%d", &how_many);
		for (int j = 1; j <= how_many; j++) {
			int work; scanf("%d", &work);
			graph[i][N+j] = 1;
		}
	}
	int answer = solve(graph[N+M+2][N+M+2], 0, N+M+1);
	printf("%d", answer);
	return 0;
}

====================================================================

//Longest Increasing Subsequence

#include <cstdio>
#include <vector>
#include <algorithm>
using namespace std;

vector<int> vec;
vector<int> lis;
vector<pair<int, int> > real_lis;

void binary_search(int x) {
	auto where_to_put = lower_bound(lis.begin(), lis.end(), vec[x - 1]);
	real_lis.push_back({where_to_put - lis.begin(), vec[x-1]});
	if ((int)lis.size() == 0) {
		lis.push_back(vec[x - 1]);
		return;
	}
	if (where_to_put - lis.begin() > (int) lis.size() - 1) {
		lis.push_back(vec[x - 1]);
	} else {
		*where_to_put = vec[x - 1];
	}
	return;
}

int main() {
	while (true) {
		int a;
		if (scanf("%d", &a) == EOF) {
			break;
		}
		vec.push_back(a);
	}
	for (int i = 1; i <= (int) vec.size(); i++) {
		binary_search(i);
	}
	printf("%d\n-\n", (int) lis.size());

	int length = (int) lis.size();
	length--;
	vector<int> realreal_lis;
	for (int i = (int) real_lis.size(); i >= 1; i--) {
		if (real_lis[i-1].first == length) {
			realreal_lis.push_back(real_lis[i-1].second);
			length--;
		}
	}
	for (int i = (int) realreal_lis.size(); i >= 1; i--) {
		printf("%d\n", realreal_lis[i-1]);
	}
	return 0;
}

====================================================================

//Dinic's Algorithm

#include <bits/stdc++.h>
using namespace std;
int w[505][505];
int ans;
int a[505];
int c[505][505];
int n;
int check[505];
int level[505];
int iter[505];
int dfs(int v,int f){
    if(v==n+1)return f;

    for(int i=iter[v] ; i<=n+1 ; iter[v]=++i){

        if(level[i]>level[v] && c[v][i]>0){
            int ret=dfs(i,min(c[v][i],f));
            if(ret){
                c[v][i]-=ret;
                c[i][v]+=ret;
                return ret;
            }
        }
    }
    return 0;
}

void bfs(){
    int i,j;
    for(i=0 ; i<=n+1 ; i++){
        level[i]=0;
        iter[i]=0;
    }
    queue<int> Q;
    Q.push(0);
    while(!Q.empty()){
        int v=Q.front();
        Q.pop();
        for(i=1 ; i<=n+1 ; i++){
            if(c[v][i] && level[i]==0){
                level[i]=level[v]+1;
                Q.push(i);
            }
        }
    }
}
int main(){
    int i,j;
    cin.tie(0);
    cout.tie(0);
    ios_base::sync_with_stdio(0);
    cin>>n;
    for(i=1  ; i<=n ; i++)cin>>a[i];
    for(i=1 ; i <=n ; i++){
        for(j=1 ; j<=n ; j++){
            cin>>w[i][j];
            c[i][j]=w[i][j];

        }
        if(a[i]==1){
            c[0][i]=c[i][0]=1e9;
        }
        if(a[i]==2){
            c[i][n+1]=c[n+1][i]=1e9;
        }
    }

    while(1){
        bfs();
        if(level[n+1]==0)break;
        int flow;
        do{
            flow=dfs(0,1e9);
            ans+=flow;
        }while(flow);
    }
    cout<<ans<<endl;

    bfs();
    for(i=1 ; i<=n ; i++)if(level[i])cout<<i<<" ";
    cout<<endl;
    for(i=1 ; i<=n ; i++)if(!level[i])cout<<i<<" ";
    cout<<endl;
    return 0;
}

//Hopcroft-Karp

#include <bits/stdc++.h>
using namespace std;
const int MX=1e6+5;
vector<int> c[MX];
int a[MX];
int b[MX];
int n;
int m;
int visit[MX];
int level[MX];

void bfs(){
    int i;
    queue<int> Q;
    for(i=1 ; i<=n ; i++){
        level[i]=0;
        if(!visit[i])Q.push(i);
    }
    while(!Q.empty()){
        int v=Q.front();    Q.pop();
        for(auto &u: c[v]){
            if(b[u] && level[b[u]]==0){
                level[b[u]]=level[v]+1;
                Q.push(b[u]);
            }
        }
    }
}

bool dfs(int v){
    for(auto &u: c[v]){
        if(b[u]==0 || level[b[u]]==level[v]+1 && dfs(b[u])){
            visit[v]=1;
            a[v]=u;
            b[u]=v;
            return 1;
        }
    }
    return 0;
}
int main(){
    cin.tie(0);
    cout.tie(0);
    ios_base::sync_with_stdio(0);
    cin>>n;
    int i;
    for(i=1 ; i<=n ; i++){
        int x,y;
        cin>>x>>y;
        c[i].push_back(x);
        c[i].push_back(y);
    }

    while(1){
        bfs();
        int flow=0;
        for(i=1 ; i<=n ; i++)
            if(!visit[i] && dfs(i))flow++;
        if(flow==0)break;
        m+=flow;
    }
    if(m<n)cout<<-1;
    else for(i=1 ; i<=n ; i++)cout<<a[i]<<"\n";
    return 0;
}

//SPFA

#include <bits/stdc++.h>
using namespace std;
int c[805][805];
int w[805][805];
int dist[805];
int inque[805];
int p[805];
int n,m;
int flow,ans;


int main(){
    cin.tie(0);
    cout.tie(0);
    ios_base::sync_with_stdio(0);
    cin>>n>>m;
    int i,j;
    for(i=1 ; i<=m ; i++)c[n+i][n+m+1]=1;
    for(i=1 ; i<=n ; i++){
        int s;
        cin>>s;
        c[0][i]=1;
        for(j=0 ; j<s ; j++){
            int x,y;
            cin>>x>>y;
            c[i][n+x]=1;
            w[i][n+x]=y;
            w[n+x][i]=-y;
        }
    }

    while(1){
        for(i=1 ; i<=n+m+1 ; i++)dist[i]=1e9;
        queue<int> Q;
        Q.push(0);
        inque[0]=1;
        while(!Q.empty()){
            int v=Q.front(); Q.pop();
            inque[v]=0;
            for(i=0 ; i<=n+m+1 ; i++){
                if( c[v][i] && dist[i]>dist[v]+w[v][i] ){
                    p[i]=v;
                    dist[i]=dist[v]+w[v][i];
                    if(!inque[i]){
                        Q.push(i);
                        inque[i]=1;
                    }
                }
            }
        }
        if(dist[n+m+1]==1e9)break;
        flow++;
        for(i=n+m+1 ; i ; i=j){
            j=p[i];
            ans+=w[j][i];
            c[j][i]--;
            c[i][j]++;
        }
    }
    cout<<flow<<endl<<ans;
    return 0;
}
==========
// Tarjan's algorithm for SCC
// foundat: analogous to time at which the vertex was discovered
// disc: will contain the foundat value of ith vertex(as in input graph)
// low: will contain the lowest vertex(foundat value) reachable from ith vertex(as in input graph)
// onstack: whether the vertex is on the stack st or not
// scc: will contain vectors of strongly connected vertices
// which can be iterated using
// for(auto i:scc){ // here i is a set of strongly connected component
//     for(auto j:i){ 
//         // iterate over the vertices in i
//     }
// }
typedef vector<int> vi;
typedef vector<vi> vvi;
#define pb push_back
#define MAX 100005

int n,m,foundat=1;
vvi graph,scc;
vi disc,low; // init disc to -1
bool onstack[MAX]; //init to 0 

void tarjan(int u){
    static stack<int> st;

    disc[u]=low[u]=foundat++;
    st.push(u);
    onstack[u]=true;
    for(auto i:graph[u]){
        if(disc[i]==-1){
            tarjan(i);
            low[u]=min(low[u],low[i]);
        }
        else if(onstack[i])
            low[u]=min(low[u],disc[i]);
    }
    if(disc[u]==low[u]){
        vi scctem;
        while(1){
            int v=st.top();
            st.pop();onstack[v]=false;
            scctem.pb(v);
            if(u==v)
                break;
        }
        scc.pb(scctem);
    }
}
int main()
{
    // n= vertices of graph
    // init
    set0(onstack);
    graph.clear();graph.resize(n+1);
    disc.clear();disc.resize(n+1,-1);
    low.clear();low.resize(n+1);
    //
    // input graph here
    FOR(i,n)
        if(disc[i+1]==-1)
            tarjan(i+1);

}


============
//Dynamic Segment Tree (Pointer-Based)

typedef long long ll;
struct Node{
    Node *l, *r; //양쪽 자식
    ll v; //구간 합
    Node(){ l = r = NULL; v = 0; }
} *root; //root 동적할당 필수!

void update(Node *node, int s, int e, int x, int v){
    if(s == e){ //리프 노드
        node->v = v; return;
    }
    int m = s + e >> 1;
    if(x <= m){
      //왼쪽 자식이 없는 경우 동적 할당
        if(!node->l) node->l = new Node();
        update(node->l, s, m, x, v);
    }else{
        //오른쪽 자식이 없는 경우 동적 할당
        if(!node->r) node->r = new Node();
        update(node->r, m+1, e, x, v);
    }
    ll t1 = node->l ? node->l->v : 0;
    ll t2 = node->r ? node->r->v : 0;
    node->v = t1 + t2;
}
ll query(Node *node, int s, int e, int l, int r){
    if(!node) return 0; //없는 노드
    if(r < s || e < l) return 0;
    if(l <= s && e <= r) return node->v;
    int m = s + e >> 1;
    return query(node->l, s, m, l, r) + query(node->r, m+1, e, l, r);
}

============
//Dynamic Segment Tree(Index-Based)

typedef long long ll;
struct Node{
    int l, r; //양쪽 자식 정점 인덱스
    ll v; //구간 합
    Node(){ l = r = -1; v = 0; }
};
Node nd[4040404]; //적당한 양 할당
//nd[0]를 루트로 잡자.
int pv = 1; //현재 pv개의 정점을 사용했음

void update(int node, int s, int e, int x, int v){
    if(s == e){
        nd[node].v = v; return;
    }
    int m = s + e >> 1;
    if(x <= m){
        if(nd[node].l == -1) nd[node].l = pv++;
        update(nd[node].l, s, m, x, v);
    }else{
        if(nd[node].r == -1) nd[node].r = pv++;
        update(nd[node].r, m+1, e, x, v);
    }
    ll t1 = nd[node].l != -1 ? nd[nd[node].l].v : 0;
    ll t2 = nd[node].r != -1 ? nd[nd[node].r].v : 0;
    nd[node].v = t1 + t2;
}

ll query(int node, int s, int e, int l, int r){
    if(node == -1) return 0;
    if(r < s || e < l) return 0;
    if(l <= s && e <= r) return nd[node].v;
    int m = s + e >> 1;
    return query(nd[node].l, s, m, l, r) + query(nd[node].r, m+1, e, l, r);
}

============

//Persistent Segment Tree

struct Node{
    Node *l, *r;
    ll v;
    Node(){ l = r = NULL; v = 0; }
};

//root[i] = i번째 세그먼트 트리의 루트
Node *root[101010]; //root[0] 할당 필수
int arr[101010]; //초깃값

void build(Node *node, int s, int e){ //0번 트리 생성
    if(s == e){
        node->v = arr[s]; return;
    }
    int m = s + e >> 1;
    node->l = new Node(); node->r = new Node();
    build(node->l, s, m); build(node->r, m+1, e);
    node->v = node->l->v + node->r->v;
}
void add(Node *prv, Node *now, int s, int e, int x, int v){
    if(s == e){
        now->v = v; return;
    }
    int m = s + e >> 1;
    if(x <= m){ //왼쪽 자식에 업데이트 하는 경우
        //왼쪽 자식은 새로운 정점 생성, 오른쪽 자식은 재활용
        now->l = new Node(); now->r = prv->r;
        add(prv->l, now->l, s, m, x, v);
    }else{
        //오른쪽 자식은 새로운 정점 생송, 왼쪽 자식은 재활용
        now->l = prv->l; now->r = new Node();
        add(prv->r, now->r, m+1, e, x, v);
    }
    now->v = now->l->v + now->r->v;
}
ll query(Node *node, int s, int e, int l, int r){
    if(r < s || e < l) return 0;
    if(l <= s && e <= r) return node->v;
    int m = s + e >> 1;
    return query(node->l, s, m, l, r) + query(node->r, m+1, e, l, r);
}
==================
//Convex Hull Trick - LineContainer
struct Line {
	mutable ll k, m, p;
	bool operator<(const Line& o) const { return k < o.k; }
	bool operator<(ll x) const { return p < x; }
};

struct LineContainer : multiset<Line, less<>> {
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	static const ll inf = LLONG_MAX;
	ll div(ll a, ll b) { // floored division
		return a / b - ((a ^ b) < 0 && a % b); }
	bool isect(iterator x, iterator y) {
		if (y == end()) return x->p = inf, 0;
		if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
		else x->p = div(y->m - x->m, x->k - y->k);
		return x->p >= y->p;
	}
	void add(ll k, ll m) {
		auto z = insert({k, m, 0}), y = z++, x = y;
		while (isect(y, z)) z = erase(z);
		if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
		while ((y = x) != begin() && (--x)->p >= y->p)
			isect(x, erase(y));
	}
	ll query(ll x) {
		assert(!empty());
		auto l = *lower_bound(x);
		return l.k * x + l.m;
	}
};
============
//RMQ
template<class T>
struct RMQ {
	vector<vector<T>> jmp;
	RMQ(const vector<T>& V) : jmp(1, V) {
		for (int pw = 1, k = 1; pw * 2 <= sz(V); pw *= 2, ++k) {
			jmp.emplace_back(sz(V) - pw * 2 + 1);
			rep(j,0,sz(jmp[k]))
				jmp[k][j] = min(jmp[k - 1][j], jmp[k - 1][j + pw]);
		}
	}
	T query(int a, int b) {
		assert(a < b); // or return inf if a == b
		int dep = 31 - __builtin_clz(b - a);
		return min(jmp[dep][a], jmp[dep][b - (1 << dep)]);
	}
};


====================
//Berlekamp-Massey

const int mod = 998244353;
using lint = long long;
lint ipow(lint x, lint p){
	lint ret = 1, piv = x;
	while(p){
		if(p & 1) ret = ret * piv % mod;
		piv = piv * piv % mod;
		p >>= 1;
	}
	return ret;
}
vector<int> berlekamp_massey(vector<int> x){
	vector<int> ls, cur;
	int lf, ld;
	for(int i=0; i<x.size(); i++){
		lint t = 0;
		for(int j=0; j<cur.size(); j++){
			t = (t + 1ll * x[i-j-1] * cur[j]) % mod;
		}
		if((t - x[i]) % mod == 0) continue;
		if(cur.empty()){
			cur.resize(i+1);
			lf = i;
			ld = (t - x[i]) % mod;
			continue;
		}
		lint k = -(x[i] - t) * ipow(ld, mod - 2) % mod;
		vector<int> c(i-lf-1);
		c.push_back(k);
		for(auto &j : ls) c.push_back(-j * k % mod);
		if(c.size() < cur.size()) c.resize(cur.size());
		for(int j=0; j<cur.size(); j++){
			c[j] = (c[j] + cur[j]) % mod;
		}
		if(i-lf+(int)ls.size()>=(int)cur.size()){
			tie(ls, lf, ld) = make_tuple(cur, i, (t - x[i]) % mod);
		}
		cur = c;
	}
	for(auto &i : cur) i = (i % mod + mod) % mod;
	return cur;
}
int get_nth(vector<int> rec, vector<int> dp, lint n){
	int m = rec.size();
	vector<int> s(m), t(m);
	s[0] = 1;
	if(m != 1) t[1] = 1;
	else t[0] = rec[0];
	auto mul = [&rec](vector<int> v, vector<int> w){
		int m = v.size();
		vector<int> t(2 * m);
		for(int j=0; j<m; j++){
			for(int k=0; k<m; k++){
				t[j+k] += 1ll * v[j] * w[k] % mod;
				if(t[j+k] >= mod) t[j+k] -= mod;
			}
		}
		for(int j=2*m-1; j>=m; j--){
			for(int k=1; k<=m; k++){
				t[j-k] += 1ll * t[j] * rec[k-1] % mod;
				if(t[j-k] >= mod) t[j-k] -= mod;
			}
		}
		t.resize(m);
		return t;
	};
	while(n){
		if(n & 1) s = mul(s, t);
		t = mul(t, t);
		n >>= 1;
	}
	lint ret = 0;
	for(int i=0; i<m; i++) ret += 1ll * s[i] * dp[i] % mod;
	return ret % mod;
}
int guess_nth_term(vector<int> x, lint n){
	if(n < x.size()) return x[n];
	vector<int> v = berlekamp_massey(x);
	if(v.empty()) return 0;
	return get_nth(v, x, n);
}
struct elem{int x, y, v;}; // A_(x, y) <- v, 0-based. no duplicate please..
vector<int> get_min_poly(int n, vector<elem> M){
	// smallest poly P such that A^i = sum_{j < i} {A^j \times P_j}
	vector<int> rnd1, rnd2;
	mt19937 rng(0x14004);
	auto randint = [&rng](int lb, int ub){
		return uniform_int_distribution<int>(lb, ub)(rng);
	};
	for(int i=0; i<n; i++){
		rnd1.push_back(randint(1, mod - 1));
		rnd2.push_back(randint(1, mod - 1));
	}
	vector<int> gobs;
	for(int i=0; i<2*n+2; i++){
		int tmp = 0;
		for(int j=0; j<n; j++){
			tmp += 1ll * rnd2[j] * rnd1[j] % mod;
			if(tmp >= mod) tmp -= mod;
		}
		gobs.push_back(tmp);
		vector<int> nxt(n);
		for(auto &i : M){
			nxt[i.x] += 1ll * i.v * rnd1[i.y] % mod;
			if(nxt[i.x] >= mod) nxt[i.x] -= mod;
		}
		rnd1 = nxt;
	}
	auto sol = berlekamp_massey(gobs);
	reverse(sol.begin(), sol.end());
	return sol;
}
lint det(int n, vector<elem> M){
	vector<int> rnd;
	mt19937 rng(0x14004);
	auto randint = [&rng](int lb, int ub){
		return uniform_int_distribution<int>(lb, ub)(rng);
	};
	for(int i=0; i<n; i++) rnd.push_back(randint(1, mod - 1));
	for(auto &i : M){
		i.v = 1ll * i.v * rnd[i.y] % mod;
	}
	auto sol = get_min_poly(n, M)[0];
	if(n % 2 == 0) sol = mod - sol;
	for(auto &i : rnd) sol = 1ll * sol * ipow(i, mod - 2) % mod;
	return sol;
}

==============
//Fast-Fourier Transform
typedef complex<double> C;
typedef vector<double> vd;
void fft(vector<C>& a) {
	int n = sz(a), L = 31 - __builtin_clz(n);
	static vector<complex<long double>> R(2, 1);
	static vector<C> rt(2, 1);  // (^ 10% faster if double)
	for (static int k = 2; k < n; k *= 2) {
		R.resize(n); rt.resize(n);
		auto x = polar(1.0L, acos(-1.0L) / k);
		rep(i,k,2*k) rt[i] = R[i] = i&1 ? R[i/2] * x : R[i/2];
	}
	vi rev(n);
	rep(i,0,n) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
	rep(i,0,n) if (i < rev[i]) swap(a[i], a[rev[i]]);
	for (int k = 1; k < n; k *= 2)
		for (int i = 0; i < n; i += 2 * k) rep(j,0,k) {
			// C z = rt[j+k] * a[i+j+k]; // (25% faster if hand-rolled)  /// include-line
			auto x = (double *)&rt[j+k], y = (double *)&a[i+j+k];        /// exclude-line
			C z(x[0]*y[0] - x[1]*y[1], x[0]*y[1] + x[1]*y[0]);           /// exclude-line
			a[i + j + k] = a[i + j] - z;
			a[i + j] += z;
		}
}
vd conv(const vd& a, const vd& b) {
	if (a.empty() || b.empty()) return {};
	vd res(sz(a) + sz(b) - 1);
	int L = 32 - __builtin_clz(sz(res)), n = 1 << L;
	vector<C> in(n), out(n);
	copy(all(a), begin(in));
	rep(i,0,sz(b)) in[i].imag(b[i]);
	fft(in);
	for (C& x : in) x *= x;
	rep(i,0,n) out[i] = in[-i & (n - 1)] - conj(in[i]);
	fft(out);
	rep(i,0,sz(res)) res[i] = imag(out[i]) / (4 * n);
	return res;
}
typedef vector<ll> vl;
template<int M> vl convMod(const vl &a, const vl &b) {
	if (a.empty() || b.empty()) return {};
	vl res(sz(a) + sz(b) - 1);
	int B=32-__builtin_clz(sz(res)), n=1<<B, cut=int(sqrt(M));
	vector<C> L(n), R(n), outs(n), outl(n);
	rep(i,0,sz(a)) L[i] = C((int)a[i] / cut, (int)a[i] % cut);
	rep(i,0,sz(b)) R[i] = C((int)b[i] / cut, (int)b[i] % cut);
	fft(L), fft(R);
	rep(i,0,n) {
		int j = -i & (n - 1);
		outl[j] = (L[i] + conj(L[j])) * R[i] / (2.0 * n);
		outs[j] = (L[i] - conj(L[j])) * R[i] / (2.0 * n) / 1i;
	}
	fft(outl), fft(outs);
	rep(i,0,sz(res)) {
		ll av = ll(real(outl[i])+.5), cv = ll(imag(outs[i])+.5);
		ll bv = ll(imag(outl[i])+.5) + ll(real(outs[i])+.5);
		res[i] = ((av % M * cut + bv) % M * cut + cv) % M;
	}
	return res;
}

==============

//Number-Theoretic Transform
const ll mod = 1000000007; // faster if const

ll modpow(ll b, ll e) {
	ll ans = 1;
	for (; e; b = b * b % mod, e /= 2)
		if (e & 1) ans = ans * b % mod;
	return ans;
}

const ll mod = (119 << 23) + 1, root = 62; // = 998244353
// For p < 2^30 there is also e.g. 5 << 25, 7 << 26, 479 << 21
// and 483 << 21 (same root). The last two are > 10^9.
typedef vector<ll> vl;
void ntt(vl &a) {
	int n = sz(a), L = 31 - __builtin_clz(n);
	static vl rt(2, 1);
	for (static int k = 2, s = 2; k < n; k *= 2, s++) {
		rt.resize(n);
		ll z[] = {1, modpow(root, mod >> s)};
		rep(i,k,2*k) rt[i] = rt[i / 2] * z[i & 1] % mod;
	}
	vi rev(n);
	rep(i,0,n) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
	rep(i,0,n) if (i < rev[i]) swap(a[i], a[rev[i]]);
	for (int k = 1; k < n; k *= 2)
		for (int i = 0; i < n; i += 2 * k) rep(j,0,k) {
			ll z = rt[j + k] * a[i + j + k] % mod, &ai = a[i + j];
			a[i + j + k] = ai - z + (z > ai ? mod : 0);
			ai += (ai + z >= mod ? z - mod : z);
		}
}
vl conv(const vl &a, const vl &b) {
	if (a.empty() || b.empty()) return {};
	int s = sz(a) + sz(b) - 1, B = 32 - __builtin_clz(s), n = 1 << B;
	int inv = modpow(n, mod - 2);
	vl L(a), R(b), out(n);
	L.resize(n), R.resize(n);
	ntt(L), ntt(R);
	rep(i,0,n) out[-i & (n - 1)] = (ll)L[i] * R[i] % mod * inv % mod;
	ntt(out);
	return {out.begin(), out.begin() + s};
}
=================
//Geometry
template <class T> int sgn(T x) { return (x > 0) - (x < 0); }
template<class T>
struct Point {
	typedef Point P;
	T x, y;
	explicit Point(T x=0, T y=0) : x(x), y(y) {}
	bool operator<(P p) const { return tie(x,y) < tie(p.x,p.y); }
	bool operator==(P p) const { return tie(x,y)==tie(p.x,p.y); }
	P operator+(P p) const { return P(x+p.x, y+p.y); }
	P operator-(P p) const { return P(x-p.x, y-p.y); }
	P operator*(T d) const { return P(x*d, y*d); }
	P operator/(T d) const { return P(x/d, y/d); }
	T dot(P p) const { return x*p.x + y*p.y; }
	T cross(P p) const { return x*p.y - y*p.x; }
	T cross(P a, P b) const { return (a-*this).cross(b-*this); }
	T dist2() const { return x*x + y*y; }
	double dist() const { return sqrt((double)dist2()); }
	// angle to x-axis in interval [-pi, pi]
	double angle() const { return atan2(y, x); }
	P unit() const { return *this/dist(); } // makes dist()=1
	P perp() const { return P(-y, x); } // rotates +90 degrees
	P normal() const { return perp().unit(); }
	// returns point rotated 'a' radians ccw around the origin
	P rotate(double a) const {
		return P(x*cos(a)-y*sin(a),x*sin(a)+y*cos(a)); }
	friend ostream& operator<<(ostream& os, P p) {
		return os << "(" << p.x << "," << p.y << ")"; }
};

template<class P>
double lineDist(const P& a, const P& b, const P& p) {
	return (double)(b-a).cross(p-a)/(b-a).dist();
}

typedef Point<double> P;
double segDist(P& s, P& e, P& p) {
	if (s==e) return (p-s).dist();
	auto d = (e-s).dist2(), t = min(d,max(.0,(p-s).dot(e-s)));
	return ((p-s)*d-(e-s)*t).dist()/d;
}

template<class P> bool onSegment(P s, P e, P p) {
	return p.cross(s, e) == 0 && (s - p).dot(e - p) <= 0;
}

template<class P> vector<P> segInter(P a, P b, P c, P d) {
	auto oa = c.cross(d, a), ob = c.cross(d, b),
	     oc = a.cross(b, c), od = a.cross(b, d);
	// Checks if intersection is single non-endpoint point.
	if (sgn(oa) * sgn(ob) < 0 && sgn(oc) * sgn(od) < 0)
		return {(a * ob - b * oa) / (ob - oa)};
	set<P> s;
	if (onSegment(c, d, a)) s.insert(a);
	if (onSegment(c, d, b)) s.insert(b);
	if (onSegment(a, b, c)) s.insert(c);
	if (onSegment(a, b, d)) s.insert(d);
	return {all(s)};
}

template<class P>
pair<int, P> lineInter(P s1, P e1, P s2, P e2) {
	auto d = (e1 - s1).cross(e2 - s2);
	if (d == 0) // if parallel
		return {-(s1.cross(e1, s2) == 0), P(0, 0)};
	auto p = s2.cross(e1, e2), q = s2.cross(e2, s1);
	return {1, (s1 * p + e1 * q) / d};
}

template<class T>
T polygonArea2(vector<Point<T>>& v) {
	T a = v.back().cross(v[0]);
	rep(i,0,sz(v)-1) a += v[i].cross(v[i+1]);
	return a;
}


typedef Point<double> P;
P polygonCenter(const vector<P>& v) {
	P res(0, 0); double A = 0;
	for (int i = 0, j = sz(v) - 1; i < sz(v); j = i++) {
		res = res + (v[i] + v[j]) * v[j].cross(v[i]);
		A += v[j].cross(v[i]);
	}
	return res / A / 3;
}

typedef Point<double> P;
vector<P> polygonCut(const vector<P>& poly, P s, P e) {
	vector<P> res;
	rep(i,0,sz(poly)) {
		P cur = poly[i], prev = i ? poly[i-1] : poly.back();
		bool side = s.cross(e, cur) < 0;
		if (side != (s.cross(e, prev) < 0))
			res.push_back(lineInter(s, e, cur, prev).second);
		if (side)
			res.push_back(cur);
	}
	return res;
}


typedef Point<ll> P;
vector<P> convexHull(vector<P> pts) {
	if (sz(pts) <= 1) return pts;
	sort(all(pts));
	vector<P> h(sz(pts)+1);
	int s = 0, t = 0;
	for (int it = 2; it--; s = --t, reverse(all(pts)))
		for (P p : pts) {
			while (t >= s + 2 && h[t-2].cross(h[t-1], p) <= 0) t--;
			h[t++] = p;
		}
	return {h.begin(), h.begin() + t - (t == 2 && h[0] == h[1])};
}

typedef Point<ll> P;
pair<P, P> closest(vector<P> v) {
	assert(sz(v) > 1);
	set<P> S;
	sort(all(v), [](P a, P b) { return a.y < b.y; });
	pair<ll, pair<P, P>> ret{LLONG_MAX, {P(), P()}};
	int j = 0;
	for (P p : v) {
		P d{1 + (ll)sqrt(ret.first), 0};
		while (v[j].y <= p.y - d.x) S.erase(v[j++]);
		auto lo = S.lower_bound(p - d), hi = S.upper_bound(p + d);
		for (; lo != hi; ++lo)
			ret = min(ret, {(*lo - p).dist2(), {*lo, p}});
		S.insert(p);
	}
	return ret.second;
}

typedef Point<ll> P;
array<P, 2> hullDiameter(vector<P> S) {
	int n = sz(S), j = n < 2 ? 0 : 1;
	pair<ll, array<P, 2>> res({0, {S[0], S[0]}});
	rep(i,0,j)
		for (;; j = (j + 1) % n) {
			res = max(res, {(S[i] - S[j]).dist2(), {S[i], S[j]}});
			if ((S[(j + 1) % n] - S[j]).cross(S[i + 1] - S[i]) >= 0)
				break;
		}
	return res.second;
}

template<class P>
int sideOf(P s, P e, P p) { return sgn(s.cross(e, p)); }

template<class P>
int sideOf(const P& s, const P& e, const P& p, double eps) {
	auto a = (e-s).cross(p-s);
	double l = (e-s).dist()*eps;
	return (a > l) - (a < -l);
}


typedef Point<ll> P;

bool inHull(const vector<P>& l, P p, bool strict = true) {
	int a = 1, b = sz(l) - 1, r = !strict;
	if (sz(l) < 3) return r && onSegment(l[0], l.back(), p);
	if (sideOf(l[0], l[a], l[b]) > 0) swap(a, b);
	if (sideOf(l[0], l[a], p) >= r || sideOf(l[0], l[b], p)<= -r)
		return false;
	while (abs(a - b) > 1) {
		int c = (a + b) / 2;
		(sideOf(l[0], l[c], p) > 0 ? b : a) = c;
	}
	return sgn(l[a].cross(l[b], p)) < r;
}

#define cmp(i,j) sgn(dir.perp().cross(poly[(i)%n]-poly[(j)%n]))
#define extr(i) cmp(i + 1, i) >= 0 && cmp(i, i - 1 + n) < 0
template <class P> int extrVertex(vector<P>& poly, P dir) {
	int n = sz(poly), lo = 0, hi = n;
	if (extr(0)) return 0;
	while (lo + 1 < hi) {
		int m = (lo + hi) / 2;
		if (extr(m)) return m;
		int ls = cmp(lo + 1, lo), ms = cmp(m + 1, m);
		(ls < ms || (ls == ms && ls == cmp(lo, m)) ? hi : lo) = m;
	}
	return lo;
}

#define cmpL(i) sgn(a.cross(poly[i], b))
template <class P>
array<int, 2> lineHull(P a, P b, vector<P> poly) {
	int endA = extrVertex(poly, (a - b).perp());
	int endB = extrVertex(poly, (b - a).perp());
	if (cmpL(endA) < 0 || cmpL(endB) > 0)
		return {-1, -1};
	array<int, 2> res;
	rep(i,0,2) {
		int lo = endB, hi = endA, n = sz(poly);
		while ((lo + 1) % n != hi) {
			int m = ((lo + hi + (lo < hi ? 0 : n)) / 2) % n;
			(cmpL(m) == cmpL(endB) ? lo : hi) = m;
		}
		res[i] = (lo + !cmpL(hi)) % n;
		swap(endA, endB);
	}
	if (res[0] == res[1]) return {res[0], -1};
	if (!cmpL(res[0]) && !cmpL(res[1]))
		switch ((res[0] - res[1] + sz(poly) + 1) % sz(poly)) {
			case 0: return {res[0], res[0]};
			case 2: return {res[1], res[1]};
		}
	return res;
}

=========
//kdTree (with Point.h)
typedef long long T;
typedef Point<T> P;
const T INF = numeric_limits<T>::max();

bool on_x(const P& a, const P& b) { return a.x < b.x; }
bool on_y(const P& a, const P& b) { return a.y < b.y; }

struct Node {
	P pt; // if this is a leaf, the single point in it
	T x0 = INF, x1 = -INF, y0 = INF, y1 = -INF; // bounds
	Node *first = 0, *second = 0;

	T distance(const P& p) { // min squared distance to a point
		T x = (p.x < x0 ? x0 : p.x > x1 ? x1 : p.x);
		T y = (p.y < y0 ? y0 : p.y > y1 ? y1 : p.y);
		return (P(x,y) - p).dist2();
	}

	Node(vector<P>&& vp) : pt(vp[0]) {
		for (P p : vp) {
			x0 = min(x0, p.x); x1 = max(x1, p.x);
			y0 = min(y0, p.y); y1 = max(y1, p.y);
		}
		if (vp.size() > 1) {
			// split on x if width >= height (not ideal...)
			sort(all(vp), x1 - x0 >= y1 - y0 ? on_x : on_y);
			// divide by taking half the array for each child (not
			// best performance with many duplicates in the middle)
			int half = sz(vp)/2;
			first = new Node({vp.begin(), vp.begin() + half});
			second = new Node({vp.begin() + half, vp.end()});
		}
	}
};

struct KDTree {
	Node* root;
	KDTree(const vector<P>& vp) : root(new Node({all(vp)})) {}

	pair<T, P> search(Node *node, const P& p) {
		if (!node->first) {
			// uncomment if we should not find the point itself:
			// if (p == node->pt) return {INF, P()};
			return make_pair((p - node->pt).dist2(), node->pt);
		}

		Node *f = node->first, *s = node->second;
		T bfirst = f->distance(p), bsec = s->distance(p);
		if (bfirst > bsec) swap(bsec, bfirst), swap(f, s);

		// search closest side first, other side if needed
		auto best = search(f, p);
		if (bsec < best.first)
			best = min(best, search(s, p));
		return best;
	}

	// find nearest point to a point, and its squared distance
	// (requires an arbitrary operator< for Point)
	pair<T, P> nearest(const P& p) {
		return search(root, p);
	}
};

============
//Fast Delaunay(with Point.h)
typedef Point<ll> P;
typedef struct Quad* Q;
typedef __int128_t lll; // (can be ll if coords are < 2e4)
P arb(LLONG_MAX,LLONG_MAX); // not equal to any other point

struct Quad {
	bool mark; Q o, rot; P p;
	P F() { return r()->p; }
	Q r() { return rot->rot; }
	Q prev() { return rot->o->rot; }
	Q next() { return r()->prev(); }
};

bool circ(P p, P a, P b, P c) { // is p in the circumcircle?
	lll p2 = p.dist2(), A = a.dist2()-p2,
	    B = b.dist2()-p2, C = c.dist2()-p2;
	return p.cross(a,b)*C + p.cross(b,c)*A + p.cross(c,a)*B > 0;
}
Q makeEdge(P orig, P dest) {
	Q q[] = {new Quad{0,0,0,orig}, new Quad{0,0,0,arb},
	         new Quad{0,0,0,dest}, new Quad{0,0,0,arb}};
	rep(i,0,4)
		q[i]->o = q[-i & 3], q[i]->rot = q[(i+1) & 3];
	return *q;
}
void splice(Q a, Q b) {
	swap(a->o->rot->o, b->o->rot->o); swap(a->o, b->o);
}
Q connect(Q a, Q b) {
	Q q = makeEdge(a->F(), b->p);
	splice(q, a->next());
	splice(q->r(), b);
	return q;
}

pair<Q,Q> rec(const vector<P>& s) {
	if (sz(s) <= 3) {
		Q a = makeEdge(s[0], s[1]), b = makeEdge(s[1], s.back());
		if (sz(s) == 2) return { a, a->r() };
		splice(a->r(), b);
		auto side = s[0].cross(s[1], s[2]);
		Q c = side ? connect(b, a) : 0;
		return {side < 0 ? c->r() : a, side < 0 ? c : b->r() };
	}

#define H(e) e->F(), e->p
#define valid(e) (e->F().cross(H(base)) > 0)
	Q A, B, ra, rb;
	int half = sz(s) / 2;
	tie(ra, A) = rec({all(s) - half});
	tie(B, rb) = rec({sz(s) - half + all(s)});
	while ((B->p.cross(H(A)) < 0 && (A = A->next())) ||
	       (A->p.cross(H(B)) > 0 && (B = B->r()->o)));
	Q base = connect(B->r(), A);
	if (A->p == ra->p) ra = base->r();
	if (B->p == rb->p) rb = base;

#define DEL(e, init, dir) Q e = init->dir; if (valid(e)) \
		while (circ(e->dir->F(), H(base), e->F())) { \
			Q t = e->dir; \
			splice(e, e->prev()); \
			splice(e->r(), e->r()->prev()); \
			e = t; \
		}
	for (;;) {
		DEL(LC, base->r(), o);  DEL(RC, base, prev());
		if (!valid(LC) && !valid(RC)) break;
		if (!valid(LC) || (valid(RC) && circ(H(RC), H(LC))))
			base = connect(RC, base->r());
		else
			base = connect(base->r(), LC->r());
	}
	return { ra, rb };
}

vector<P> triangulate(vector<P> pts) {
	sort(all(pts));  assert(unique(all(pts)) == pts.end());
	if (sz(pts) < 2) return {};
	Q e = rec(pts).first;
	vector<Q> q = {e};
	int qi = 0;
	while (e->o->F().cross(e->F(), e->p) < 0) e = e->o;
#define ADD { Q c = e; do { c->mark = 1; pts.push_back(c->p); \
	q.push_back(c->r()); c = c->next(); } while (c != e); }
	ADD; pts.clear();
	while (qi < sz(q)) if (!(e = q[qi++])->mark) ADD;
	return pts;
}
