#include <math.h>

double Pee(double Enu, double L, double Ue4_2, double deltam_2) {
    return 1 - 4 * (Ue4_2 - Ue4_2 * Ue4_2) * pow(sin(1.267 * deltam_2 * L / Enu), 2);
}

double EeeFlux(double Enu, double L, double Ue4_2, double deltam_2) {
    double mMu = 105.6583755;  // MeV
    return (192.0 / mMu) * pow(Enu / mMu, 2) * (0.5 - Enu / mMu) * Pee(Enu, L, Ue4_2, deltam_2);
}

double TtoQ(double T, double M) {
    double hbarc_fmMeV = 197.327;  // fm MeV
    return sqrt(2 * M * T + T * T) / hbarc_fmMeV;
}

double rnToR0(double rn) {
    double s = 0.3;  // fm
    return sqrt(5.0 / 3.0 * (rn * rn - 3 * s * s));
}

double j1(double x) {
    return (-x * cos(x) + sin(x)) / (x * x);
}

double helmff(double Q, double rn) {
    double s = 0.3;  // fm
    if (Q == 0) {
        return 1;
    }
    return (3 * j1(Q * rnToR0(rn)) / (Q * rnToR0(rn))) * exp(-Q * Q * s * s / 2);
}

double F(double T, double rn, double M) {
    return helmff(TtoQ(T, M), rn);
}

double XSection(double Enu, double Er) {
    double Gv = 0.0298 * 55 - 0.5117 * 78;
    double Ga = 0;
    double M = 123800.645;  // MeV
    double GF = 1.16637e-11;  // MeV^-2
    double rn = 4.77305;  // fm
    double a = 0.0749 / 1000;
    double b = 9.56 * 1000;
    double hbarc_cmMeV2 = 3.894e-22;  // cm^2 MeV^2

    return (hbarc_cmMeV2 * GF * GF * M / (2 * M_PI)) * (F(Er, rn, M) * F(Er, rn, M)) * ((Gv + Ga) * (Gv + Ga) + (Gv - Ga) * (Gv - Ga) * (1 - Er / Enu) * (1 - Er / Enu) - (Gv * Gv - Ga * Ga) * M * (Er / (Enu * Enu)));
}

double smear(double x, double e) {
    double a = 0.0749 / 1000;
    double b = 9.56 * 1000;

    return pow(a / e * (1 + b * e), 1 + b * e) / tgamma(1 + b * e) * pow(x, b * e) * exp(-a / e * (1 + b * e) * x);
}

double quenching_factor(double Erec) {
    return 0.0554628 * Erec + 4.30681 * pow(Erec, 2) - 111.707 * pow(Erec, 3) + 840.384 * pow(Erec, 4);
}

double recoil_spectrum(int n, double *x, void *user_data) {
    double Enu = x[0];
    double Er = x[1];

    double *theta = (double *)user_data;

    double pe = theta[0];

    return EeeFlux(Enu, 19.3, theta[1], theta[2]) * XSection(Enu, Er) * smear(pe, quenching_factor(Er));
}

double recoil_spectrum_int(int n, double *x, void *user_data) {
    double Enu = x[0];
    double Er = x[1];
    double pe = x[2];

    double *theta = (double *)user_data;

    return EeeFlux(Enu, 19.3, theta[0], theta[1]) * XSection(Enu, Er) * smear(pe, quenching_factor(Er));
}