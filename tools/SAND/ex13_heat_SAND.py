from pyfreefem import FreeFemRunner
from pymedit import P1Function
from nullspace_optimizer import Optimizable, memoize, nlspace_solve

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


# Some constants
CONST_Q = 1e4
CONST_KAPPA_F = 401
CONST_KAPPA_S = 1
SIMP_P = 3

def init_mesh(n=50):
    mesh_script = """   
        IMPORT "io.edp"
        mesh Th = square($n, $n, flags=1);  
        exportMesh(Th);"""
    Th = FreeFemRunner(mesh_script).execute({'n': n})['Th']

    # Changing boundary label on left hand side
    pts1 = Th.vertices[Th.edges[:, 0]-1]
    pts2 = Th.vertices[Th.edges[:, 1]-1]
    eps = 1/(10*n)
    indices = np.logical_and(
        abs(pts1[:, 1]-0.5) <= 0.1+eps, abs(pts2[:, 1]-0.5) <= 0.1+eps)
    indices = np.logical_and(indices, Th.edges[:, -1] == 4)
    Th.edges[indices, -1] = 5
    Th.edges[Th.edges[:, -1] != 5, -1] = 1

    return Th

def init_filter_matrix(Th, gamma=2):
    filter_script = """ 
    IMPORT "io.edp" 
    mesh Th = importMesh("Th");
    
    fespace Fh1(Th,P1);

    macro grad(u) [dx(u),dy(u)]//
    
    real gamma = $gamma * Th.hmin;
    varf helmholtz(u,v) = int2d(Th)(gamma^2*grad(u)'*grad(v)+u*v);
    matrix A = helmholtz(Fh1,Fh1);
    
    exportMatrix(A);
    """

    runner = FreeFemRunner(filter_script)
    runner.import_variables(Th=Th)
    exports = runner.execute({"gamma":gamma})
    A = exports["A"]

    return A

@memoize()
def retract_state(Th, rho, rescale=1.):
    retract_script = """
    IMPORT "io.edp" 
    load "Element_P3"

    mesh Th = importMesh("Th");

    fespace Fh1(Th,P1);
    fespace Fh3(Th,P3);

    Fh1 rho;
    rho[] = importArray("rho");

    real p = $p;
    real kappaf = $kappaf;
    real kappas = $kappas;
    Fh3 kappa = rho^p*(kappaf-kappas)+kappas;

    func Q = $Q;
    macro grad(u) [dx(u),dy(u)] //

    Fh1 T, S;
    solve heat(T,S)= int2d(Th)(kappa*grad(T)'*grad(S))-int2d(Th)(Q*S) + on(5,T=0);

    exportArray(T[]);
    """
    runner = FreeFemRunner(retract_script)
    runner.import_variables(Th=Th, rho=rho)
    return runner.execute({'p':SIMP_P, 'Q':CONST_Q*rescale, 'kappaf':CONST_KAPPA_F, 'kappas':CONST_KAPPA_S})['T[]']

@memoize()
def solve_state(Th, rho, T, rescale=1):
    solve_script = """
    IMPORT "io.edp" 
    load "Element_P3"

    mesh Th = importMesh("Th");

    fespace Fh1(Th,P1);
    fespace Fh2(Th,P2);
    fespace Fh3(Th,P3);

    Fh1 rho;
    rho[] = importArray("rho");
    Fh1 T;
    T[] = importArray("T");

    // === Problem parameters === //
    real p = $p;
    real kappaf = $kappaf;
    real kappas = $kappas;
    Fh3 kappa = rho^p*(kappaf-kappas)+kappas;
    Fh2 dkappa = p*rho^(p-1)*(kappaf-kappas);
    Fh1 S;
    func Q = $Q;

    // === Macros === //
    macro grad(u) [dx(u),dy(u)] //

    // === Calculate J and DJ === //
    real vol0 = int2d(Th)(1.);
    real J = int2d(Th)(rho/(vol0));
    varf vDJr(dummy, drho) = int2d(Th)(drho/(vol0));
    real[int] DJr = vDJr(0,Fh1);

    // === Calculate K, F and G === //
    varf vArho(T, S) = int2d(Th)(kappa*grad(T)'*grad(S)) + on(5, T=0);
    matrix Arho = vArho(Fh1, Fh1, tgv=-1);
    varf vF(T, S) = int2d(Th)(Q*S) + on(5, T=0);
    Fh1 F, LHS;
    F[] = vF(0, Fh1);
    LHS[] = Arho * T[];
    real[int] G = LHS[] - F[];

    // === Calculate D_rho(KT) === //
    varf vDAr(drho, S) = int2d(Th)(dkappa*grad(T)'*grad(S)*drho) + on(5, drho=0);
    matrix DAr = vDAr(Fh1, Fh1, tgv=-10);

    exportVar(J);
    exportArray(G);

    exportArray(DJr);
    exportMatrix(DAr);   // = D_rho(G)
    exportMatrix(Arho);  // = D_T(G)
    """
    runner = FreeFemRunner(solve_script)
    runner.import_variables(Th=Th, rho=rho, T=T)
    return runner.execute({'p': SIMP_P, 'Q': CONST_Q*rescale, 'kappaf': CONST_KAPPA_F, 'kappas': CONST_KAPPA_S})


### ============================================ ###
### Local constrainst using the SAND formulation ###
### ============================================ ###
class Heat_TO_SAND(Optimizable):
    def __init__(self, n=50, vfrac0=0.4, gamma=2, maxT=300, plot=False):
        self.maxT = maxT

        # Initialize mesh
        self.Th = init_mesh(n)
        self.nbRho = self.Th.nv
        self.nbT = self.Th.nv

        # Set inner product matrix A
        I_nv = sp.eye(self.Th.nv, format="csc")
        self.rhoFilter = init_filter_matrix(self.Th, gamma)
        ida = sp.linalg.norm(self.rhoFilter)/np.sqrt(self.Th.nv)
        self.IP = sp.block_diag((self.rhoFilter, I_nv*ida), format="csc")

        # Compute initial state
        rho0 = vfrac0 * np.ones(self.nbRho, dtype=float)  
        T0 = retract_state(self.Th, rho0)
        self.rescale = 1/(maxT)
        self.rhoT0 = np.concatenate((rho0, T0*self.rescale))

        self.plot = plot 
        if self.plot:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2)
            self.fig.set_size_inches(12, 6)
            self.ax = self.ax.flatten()
            self.cbars = [None]*2

    def x0(self):
        return self.rhoT0
    
    def J(self, rhoT):
        return self.solve(rhoT)['J']
    
    def G(self, rhoT):
        return self.solve(rhoT)['G']

    def H(self, rhoT):
        rho = rhoT[:self.nbRho]
        T = rhoT[self.nbRho:]
        return np.concatenate((-rho, rho-1., T - self.maxT*self.rescale))

    def dJ(self, rhoT):
        return np.concatenate((self.solve(rhoT)['DJr'], np.zeros(self.nbT)))
    
    def dG(self, rhoT):
        exports = self.solve(rhoT)
        dGr = exports['DAr']
        dGT = exports['Arho']
        return sp.hstack((dGr , dGT), format="csc")

    def dH(self, rhoT):
        I = sp.eye(self.nbRho, format="csc")

        DHr = sp.vstack((-I, I, sp.csc_matrix((self.nbT, self.nbRho))))
        DHT = sp.vstack((sp.csc_matrix((2*self.nbRho, self.nbT)), I))
        return sp.hstack((DHr,DHT))
    
    def inner_product(self, rhoT):
        return self.IP
    
    def retract(self, rhoT, drhoT):
        new_rhoT = rhoT + drhoT
        new_rhoT[:self.nbRho] = np.maximum(np.minimum(new_rhoT[:self.nbRho], 1), 0)
        new_rhoT[self.nbRho:] = retract_state(self.Th, new_rhoT[:self.nbRho], self.rescale)

        return new_rhoT
    
    def accept(self, params, results):
        if self.plot:
            rhoT = results['x'][-1]
            self.show_state(rhoT, fig_title="Iteration "+str(results['it'][-1]))
        params['normalisation_norm'] = lambda rhoT : np.linalg.norm(rhoT[:self.nbRho], np.inf)
        if "maxHT" in results:
            results["maxHT"].append(np.max(self.H(results["x"][-1])[2*self.nbRho:]))
        else:
            results["maxHT"] = [np.max(self.H(results["x"][-1])[2*self.nbRho:])]

    def solve(self, rhoT):
        rho = rhoT[:self.nbRho]
        T = rhoT[self.nbRho:]
        return solve_state(self.Th, rho, T, self.rescale)

    def show_state(self, rhoT, fig_title="State (rho and T)"):
        if self.plot:
            fig = self.fig
            ax = self.ax
            cbars = self.cbars
            for i in range(2):
                if cbars[i]:
                    cbars[i].remove()
                ax[i].clear()
        else:
            plt.ion()
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)
            ax = ax.flatten()
            cbars = [None, None]

        rho = rhoT[:self.nbRho]
        T = rhoT[self.nbRho:]/self.rescale

        _, _, cbars[0] = P1Function(self.Th, rho).plot(fig=fig, ax=ax[0], cmap="gray_r", vmin=0, vmax=1)
        _, _, cbars[1] = P1Function(self.Th, T).plot(fig=fig, cmap="turbo", ax=ax[1], type_plot='tricontourf')
        ax[0].title.set_text(r'$\rho$')
        ax[1].title.set_text(r'$T$')
        fig.suptitle(fig_title)
        plt.pause(0.1)

    def show_results(self, results):
        rhoT = results["x"][-1]

        # Draw final state
        self.show_state(rhoT, "Final State")

        # Draw objective function
        from nullspace_optimizer.examples.basic_examples.utils import draw
        plt.figure().clear()
        draw.drawJ(results)

        # Draw the times
        if "time" in results:
            plt.figure().clear()
            if len(results["it"]) == len(results["time"])+1:
                plt.plot(results["it"][:-1], results["time"])
            else:
                plt.plot(results["it"], results["time"])
            plt.ylabel("s")
            time_s = np.sum(results["time"])
            plt.title("Time per iteration (s). Total: "+str(int(time_s//3600))+"h : "+str(int((time_s%3600)//60))+"m : "+str(int(time_s%60))+"s")

        # Draw the constraint qualifications
        if "maxH" in results and "||G||" in results:
            plt.figure().clear()
            plt.plot(results["it"], results["||G||"], label="||G||")
            if "maxHT" in results:
                plt.plot(results["it"], np.asarray(results["maxHT"]), label="max(T - Tmax)")
            plt.yscale("symlog", linthresh=1e-8)
            plt.gca().legend()
        
        # Print final volume fraction
        print("=== Results: ===")
        print("-> Number of variables:", len(rhoT))
        print("-> Number of equality constraints:", len(self.G(rhoT)))
        print("-> Number of inequality constraints:", len(self.H(rhoT)))
        print("-> J =", self.J(rhoT))
        print("-> ||G|| =", np.linalg.norm(self.G(rhoT)))
        print("-> max(H) =", np.max(self.H(rhoT)))


def check_problem(problem, plot=True):
    x = problem.x0()
    print("\n========= Problem: Heat SAND with exact local constraints =========")
    print("--- Some properties of the problem: ---")
    print(f" len(x) = {len(x)}")
    print(f" # G = {len(problem.G(x))}")
    print(f" # H = {len(problem.H(x))}")
    print("---------------------------------------")
    print("Reminder:")
    print(" x = [rho, T]")
    print(" J(rho, T) = int2d(Th)(rho/vol0) ")
    print(" G(rho, T) = K(rho)T - F")
    print(" H(rho, T) = [ -rho, ")
    print("               rho - 1,")
    print("               T - maxT*rescale")
    print("=======================================")

    # Since we retract to a solution of the equality constraints, we have to use a 'correct' step and not just a random one.
    # So the step must satisfy: dG xi = 0 --> dG_rho xi_rho + dG_T xi_T = 0 --> xi_T = - dG_T^-1 (dG_rho xi_rho)
    import scipy.sparse.linalg as lg
    from nullspace_optimizer.utils import check_derivatives

    drho = np.random.rand(problem.nbRho)
    dG = problem.dG(x)
    dGrho = dG[:, :problem.nbRho]
    dGTinv = lg.factorized(dG[:, problem.nbRho:])
    dT = - dGTinv(dGrho @ drho)
    dx = np.concatenate((drho, dT))

    return check_derivatives(problem, dx=dx, random_init=False, plot=plot, verbose=False)


if __name__ == "__main__":
    from nullspace_optimizer.inout import parse_args
    args = parse_args()
    N = args.pop('N', 50)
    plot = args.pop('plot', False)

    problem = Heat_TO_SAND(N, plot=plot)
    if args.pop('check', False):
        import sys
        sys.exit(check_problem(problem))

    params = dict(dt=0.05, itnormalisation=50, maxit=150,
                  save_only_N_iterations=1,
                  save_only_Q_constraints=5,
                  qp_solver='qpalm',
                  qp_solver_options = dict(max_iter=1000),
                  tol_qp=1e-8,
                  method_xiC="qp",
                  K=0.01,
                  qp_saturate_slack=True)
    params.update(args)
    results = nlspace_solve(problem, params)

    problem.show_results(results)
    input("Press enter to close the figures ...")

