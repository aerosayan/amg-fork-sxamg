
/* this example solves a Poisson equation with Dirichlet boundary condition using AMG
 * 
 * - \delta u = g
 * 
 * The exact solution is u(x,y,z) = e^(x^2 + y^2 + z^2). 
 *
 * The domain is [0,1]^3.
 *
 * To simplify the example, uniform grid is used. Each dimension is divided into N intervals.
 * The grid has N^3 elements and (N + 1)^3 vertices. 
 *
 * The dimension of linear system is (N + 1)^3 * (N + 1)^3.
 *
 * mapping:
 * (i, j, k) ==> i + j * (N + 1) + k * (N + 1)^2
 *
 * */

#include "sxamg.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* proto types */
double func_g(double x, double y, double z);
double func_b(double x, double y, double z);

void id_to_ijk(SX_INT *i, SX_INT *j, SX_INT *k, SX_INT id, SX_INT N);
SX_INT ijk_to_id(SX_INT i, SX_INT j, SX_INT k, SX_INT N);

void assemble_linear_system(SX_MAT *A, SX_VEC *b, SX_INT size, SX_INT N, double h);

void set_real_solution(SX_VEC *u, SX_INT size, SX_INT N, double h);
double compute_error(SX_VEC *u, SX_VEC *ux);

/* implementations */
double func_g(double x, double y, double z)
{
    return -(6 + 4 * x * x + 4 * y * y + 4 * z * z) * exp(x * x + y * y + z * z);
}

double func_b(double x, double y, double z)
{
    return exp(x * x + y * y + z * z);
}

void id_to_ijk(SX_INT *i, SX_INT *j, SX_INT *k, SX_INT id, SX_INT N)
{
    /* K */
    *k = id / ((N + 1) * (N + 1));

    /* J */
    id -= *k * (N + 1) * (N + 1);
    *j = id / (N + 1);

    /* I */
    *i = id - *j * (N + 1);
}

SX_INT ijk_to_id(SX_INT i, SX_INT j, SX_INT k, SX_INT N)
{
    return k * (N + 1) * (N + 1) + j * (N + 1) + i;
}

void assemble_linear_system(SX_MAT *A, SX_VEC *b, SX_INT size, SX_INT N, double h)
{
    SX_INT i;
    SX_INT si, sj, sk;
    SX_FLT *vd;
    SX_INT *Ap, *Aj;
    SX_FLT *Ax;
    SX_INT p;

    assert(A != NULL);
    assert(b != NULL);
    assert(N > 1);
    assert(size == (N + 1) * (N + 1) * (N + 1));
    assert(h > 0);

    /* vector */
    vd = b->d;

    /* matrix: each row has 7 entries at most */
    Ap = sx_malloc((size + 1) * sizeof(*Ap));
    Aj = sx_malloc(size * 7 * sizeof(*Ap));
    Ax = sx_malloc(size * 7 * sizeof(*Ax));

    A->num_rows = A->num_cols = size;
    A->Ap = Ap;
    A->Aj = Aj;
    A->Ax = Ax;

    Ap[0] = p = 0;
    for (i = 0; i < size; i++) {
        id_to_ijk(&si, &sj, &sk, i, N);

        /* boundary nodes */
        if (si == 0 || si == N || sj == 0 || sj == N || sk == 0 || sk == N) {
            /* rhs: b */
            vd[i] = func_b(si * h, sj * h, sk * h);

            /* matrix */
            Aj[p] = i;
            Ax[p] = 1;

            p++;
        }
        else { /* interior nodes */
            /* rhs: b */
            vd[i] = h * h * func_g(si * h, sj * h, sk * h);

            /* 7 entries */
            Aj[p] = ijk_to_id(si, sj, sk - 1, N);
            Ax[p] = -1;
            p++;

            Aj[p] = ijk_to_id(si, sj - 1, sk, N);
            Ax[p] = -1;
            p++;

            Aj[p] = ijk_to_id(si - 1, sj, sk, N);
            Ax[p] = -1;
            p++;

            /* diagonal */
            Aj[p] = ijk_to_id(si, sj, sk, N);
            Ax[p] = 6;
            p++;

            Aj[p] = ijk_to_id(si + 1, sj, sk, N);
            Ax[p] = -1;
            p++;

            Aj[p] = ijk_to_id(si, sj + 1, sk, N);
            Ax[p] = -1;
            p++;

            Aj[p] = ijk_to_id(si, sj, sk + 1, N);
            Ax[p] = -1;
            p++;
        }

        Ap[i + 1] = p;
    }

    /* nnz */
    A->num_nnzs = p;
}

void set_real_solution(SX_VEC *u, SX_INT size, SX_INT N, double h)
{
    SX_INT i;
    SX_INT si, sj, sk;
    SX_FLT *vd;

    assert(u != NULL);
    assert(N > 1);
    assert(size == (N + 1) * (N + 1) * (N + 1));
    assert(h > 0);

    /* vector */
    vd = u->d;

    for (i = 0; i < size; i++) {
        id_to_ijk(&si, &sj, &sk, i, N);
        vd[i] = func_b(si * h, sj * h, sk * h);
    }
}

double compute_error(SX_VEC *u, SX_VEC *ux)
{
    SX_INT i, size = u->n;
    SX_FLT *vd, *vx;
    double t = 0.;
    double nrm = 0.;

    /* vector */
    vd = u->d;
    vx = ux->d;

    for (i = 0; i < size; i++) {
        nrm += vd[i] * vd[i];
        t += (vd[i] - vx[i]) * (vd[i] - vx[i]);
    }

    return sqrt(t / nrm);
}

int main(int argc, char **argv)
{
    SX_INT N = 100;
    SX_INT size;
    double h, tm;

    SX_MAT A;
    SX_VEC b, x, u;

    SX_RTN rtn;
    SX_AMG_PARS pars;

    if (argc > 1) {
        N = atoi(argv[1]);

        if (N <= 1) {
            printf("N is too small\n");
            exit(0);
        }
    }

    assert(N > 1);
    size = (N + 1) * (N + 1) * (N + 1);

    /* rhs */
    b = sx_vec_create(size);

    /* h */
    h = 1. / N;

    /* assemble */
    tm = sx_get_time();
    assemble_linear_system(&A, &b, size, N, h);
    tm = sx_get_time() - tm;

    sx_printf("\nlinear system dimension: %"dFMT"\n", size);
    sx_printf("linear system assembling time: %g s\n", tm);

    /* solve the system */
    x = sx_vec_create(size);
    sx_vec_set_value(&x, 0.0);

    /* pars */
    sx_amg_pars_init(&pars);
    pars.maxit = 1000;
    pars.verb = 3;
    pars.max_levels = 10;
    pars.interp_type = SX_INTERP_DIR;
    pars.tol = 1e-9;

    sx_amg_pars_print(&pars);

    rtn = sx_solver_amg(&A, &x, &b, &pars);

    sx_printf("AMG residual: %"fFMTg"\n", rtn.ares);
    sx_printf("AMG relative residual: %"fFMTg"\n", rtn.rres);
    sx_printf("AMG iterations: %"dFMT"\n", rtn.nits);

    /* real solution */
    u = sx_vec_create(size);
    set_real_solution(&u, size, N, h);
    sx_printf("\nrelative error: %g\n", compute_error(&u, &x));

    sx_mat_destroy(&A);
    sx_vec_destroy(&x);
    sx_vec_destroy(&b);
    sx_vec_destroy(&u);

    return 0;
}
