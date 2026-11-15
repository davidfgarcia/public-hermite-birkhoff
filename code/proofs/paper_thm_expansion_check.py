"""Independent exact check of Theorem (exact finite expansion), eq. (finite) of the paper, in the PAPER's formalism
(R = N_i^perp with Euclidean inner product on monomial coefficients, L = Lambda_i|_R, Theta, Gamma, gamma, M=(L*L)^-1),
against the exact engine proofs/perturbative_engine.Config.direct_error (a different formalism). d=1, rho=1, y_i=0 (p_i*=0).
Run from code/: PYTHONPATH=. python3 proofs/paper_thm_expansion_check.py"""
import sys, random, math, sympy as sp
sys.path.insert(0,'.')
from proofs.perturbative_engine import Config
Qr=sp.Rational; z=sp.symbols('z')
def row(sym,x0,m):
    # functional p -> s(D)p(x0) on monomial coefficient vectors, sym = {e: coef} (xi^e -> d^e)
    r=[]
    for k in range(m+1):
        p=z**k; val=0
        for (e,),c in sym.items():
            val+=c*sp.diff(p,z,e).subs(z,x0)
        r.append(val)
    return sp.Matrix([r])
def paper_error(central, others, m, alpha, q):
    # central: list of symbols (dicts {(e,):c}) at 0; others: list of (x_j, [symbols], data list) ; q: dict {(e,):c}
    Li=sp.Matrix.vstack(*[row(s,0,m) for s in central])
    N=Li.nullspace(); Nb=sp.Matrix.hstack(*N) if N else sp.zeros(m+1,0)
    # R = N^perp (Euclidean): columns spanning rowspace of Li
    Rb=Li.T                                   # basis of R (rows of Li), L = Li restricted: coordinates c -> Li*(Li^T c)
    Lm=Li*Rb                                  # maps R-coordinates (c) to R^{n_i}
    # work in R-coordinates c w.r.t. basis Rb (not orthonormal): use the Gram matrix for the inner product
    G=Rb.T*Rb
    # B_x(p)=sum w_j |Lambda_j p - y_j|^2 ; Theta = sum w Lambda_j^T Lambda_j ; b = sum w Lambda_j^T y_j
    Th=sp.zeros(m+1,m+1); b=sp.zeros(m+1,1)
    for (xj,syms,ys) in others:
        Lj=sp.Matrix.vstack(*[row(s,xj,m) for s in syms]); w=(z-xj)**(-alpha)
        Th+=w*Lj.T*Lj; b+=w*Lj.T*sp.Matrix(ys)
    # blocks in full coordinates: P_R,P_N via bases (Euclidean orthogonal projectors)
    PR=Rb*(Rb.T*Rb).inv()*Rb.T; PN=sp.eye(m+1)-PR
    if Nb.shape[1]>0:
        Nn=Nb.T*Th*Nb; Gam=Rb.T*Th*Rb-Rb.T*Th*Nb*Nn.inv()*Nb.T*Th*Rb
        gam=Rb.T*b-Rb.T*Th*Nb*Nn.inv()*Nb.T*b
    else:
        Gam=Rb.T*Th*Rb; gam=Rb.T*b
    # in R-coordinates: <.,.> uses Gram G; (L*L) = Lm^T Lm w.r.t. coordinates -> operator on coordinates: G^-1 (Lm^T Lm)
    # the identity (L*L + t Gamma) s = gamma holds in the dual pairing; in coordinates: (Lm^T Lm + t*Gam) c = gam with Gam,gam as above (covariant form)
    A=Lm.T*Lm; Minv=A.inv()
    # q_k and k*
    qd=sum(c*z**e for (e,),c in q.items()); qs=[]; cur=sp.diff(qd,z,alpha)
    while cur!=0:
        qs.append(cur); cur=sp.diff(cur,z,alpha)
    total=0
    vec=Minv*gam
    for k in range(1,len(qs)+1):
        coeffs=Rb*vec                           # polynomial (coefficient vector in monomials)
        eps_z=sum(coeffs[e]*z**e for e in range(m+1))
        eps_z=sp.together(eps_z)
        qk=sp.Poly(qs[k-1],z)  # qs[k-1]=q_k=Delta^{k alpha/2} q
        val=sum(cf*sp.diff(eps_z,z,mon[0]).subs(z,0) for mon,cf in zip(qk.monoms(),qk.coeffs()))
        total+=(-1)**(k-1)*val
        vec=Minv*(Gam*vec)                      # next: M (Gamma M)^k gamma
    return sp.nsimplify(total)
def run(name,central,others,m,alpha,q,hdata):
    nodes=[((sp.Integer(0),),central_t(central),1)]+[((xj,),[{(e,):c for (e,),c in s.items()} for s in syms],1) for (xj,syms,_) in others]
    C=Config(nodes,m,alpha)
    ys,_=C.data_of(hdata)
    eng=C.direct_error({(e,):c for (e,),c in q.items()},ys)
    # the data vectors as lists for the paper formula
    others_d=[(xj,syms,list(y)) for (xj,syms,_),y in zip(others,ys)]
    pap=paper_error(central,others_d,m,alpha,q)
    print(f"{name}: engine={sp.nsimplify(eng)}  paper-formula={pap}  equal={sp.simplify(eng-pap)==0}")
def central_t(c): return [{(e,):v for (e,),v in s.items()} for s in c]
sym=lambda *items:{(e,):c for e,c in items}
jet=lambda r:[sym((e,1)) for e in range(r+1)]
random.seed(5)
# 1: N_i=0, k*=2
run("N_i=0, m=4, a=2, q=xi^4",jet(4),[(Qr(1),[sym((0,1))],None),(Qr(-2),[sym((0,1))],None),(Qr(3,2),[sym((0,1))],None)],4,2,sym((4,1)),{(1,):1,(3,):2})
# 2: N_i!=0 (m=5, jets 0..4), others: values+derivative at one node
run("N_i!=0, m=5, a=2, q=xi^4",jet(4),[(Qr(1),[sym((0,1))],None),(Qr(-2),[sym((0,1)),sym((1,1))],None),(Qr(3),[sym((0,1))],None)],5,2,sym((4,1)),{(0,):1,(2,):1,(5,):3})
# 3: skew central tuple
while True:
    M=sp.Matrix(5,5,lambda i,j: random.randint(-2,2))
    if M.det()!=0: break
skew=[{(e,):M[s,e] for e in range(5) if M[s,e]!=0} for s in range(5)]
run("skew tuple, N_i=0, m=4, a=2, q=xi^4+xi^2",skew,[(Qr(1),[sym((0,1))],None),(Qr(-2),[sym((0,1))],None),(Qr(3,2),[sym((0,1))],None)],4,2,sym((4,1),(2,1)),{(1,):1,(3,):2,(0,):1})
# 4: alpha=4, k*=1, N_i!=0
run("alpha=4, N_i!=0, m=6, jets 0..5, q=xi^5",jet(5),[(Qr(1),[sym((0,1))],None),(Qr(-2),[sym((0,1)),sym((1,1))],None),(Qr(3),[sym((0,1))],None)],6,4,sym((5,1)),{(0,):1,(3,):1,(6,):2})
