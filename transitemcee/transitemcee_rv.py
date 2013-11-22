import transitemcee
from transitemcee import get_ar
import numpy as np
from copy import deepcopy
from scipy.stats import truncnorm
from claretquadpy import claretquad
from claret4ppy import claretlimb4p
from numpy import random
import time as thetime
import emcee
import tmodtom as tmod
import sys

class transitemcee_rv(transitemcee.transitemcee_fitldp,transitemcee.transitemcee):

    def __init__(self,nplanets,cadence=1625.3,rvcadence=1800.0,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):
        super(transitemcee_rv,self).__init__(nplanets,cadence=cadence,
            ldfileloc=ldfileloc,
            codedir=codedir)
        sys.path.append(codedir)
        self.rvcadence = rvcadence / 86400.

    def already_open(self,t1,f1,e1,rvt1,rvval1,rverr1,
        timeoffset=0.0,rvtimeoffset=0.0,normalize=False):
        time = t1 - timeoffset
        if normalize:
            flux = f1 / np.median(f1)
            err = e1 / np.median(f1)
        else:
            flux = f1
            err = e1

        rvtime = rvt1 - rvtimeoffset
        if normalize:
            rvval = rvval1 - np.median(rvval1)
            rverr = rverr1
        else:
            rvval = rvval1
            rverr = rverr1


        self.time = time
        self.flux = flux
        self.err = err
        self.npt = len(time)
        self._itime = np.zeros(self.npt) + self.cadence
        self._datatype = np.zeros(self.npt)

        self.rvval = rvval
        self.rvtime = rvtime
        self.rverr = rverr
        self.rvnpt = len(rvtime)
        self._rvitime = np.zeros(self.rvnpt) + self.rvcadence

    def get_sol(self,*args,**kwargs):
        """
        like get_sol from regular transitemcee
        but with an extra variable for every
        planet -- rvamp
        also veloffset is impoartant now
        """
        tom = args
        assert np.shape(tom)[0] == self.nplanets * 7

        if 'dil' in kwargs.keys():
            dil = kwargs['dil']
            print ' running with dil = %s' %(dil)
        else:
            dil = 0.0

        if 'veloffset' in kwargs.keys():
            self.veloffset = kwargs['veloffset']
        else:
            self.veloffset = 0.0

        occ = 0.0
        alb = 0.0
        ell = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6

        fit_sol = np.array([self.rho_0,self.zpt_0,
            self.ld1,self.ld2,self.veloffset])

        for i in xrange(self.nplanets):
            T0_0 = args[i*7]
            per_0 = args[i*7 +1]
            b_0 = args[i*7 +2]
            rprs_0 = args[i*7 +3]
            ecosw_0 = args[i*7 +4]
            esinw_0 = args[i*7 +5]
            rvamp_0 = args[i*7 + 6]

            new_params = np.array([T0_0,per_0,
                b_0,rprs_0,ecosw_0,esinw_0,rvamp_0])
            fit_sol = np.r_[fit_sol,new_params]

        self.fit_sol = fit_sol
        self.fit_sol_0 = deepcopy(self.fit_sol)

        self.fixed_sol = np.array([
            dil,occ,ell,alb])

    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw,rvamp
        """

        rho_unc = 0.001
        zpt_unc = 1.E-8
        ld1_unc = 0.05
        ld2_unc = 0.05
        veloffset_unc = 100.
        T0_unc = 0.0002
        per_unc = 0.00005
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        rvamp_unc = 10.

        if self.n_ldparams == 2:
            p0 = np.zeros([nwalkers,5+self.nplanets*7+2])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]
        veloffset = self.fit_sol[4]

        start,stop = ((0.0001 - rho) / rho_unc,
            (30.0 - rho) / rho_unc)
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,
            size=nwalkers)

        start,stop = ((0.0 - ld1) / ld1_unc,
            (1.0 - ld1) / ld1_unc)
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = ((0.0 - ld2) / ld2_unc,
            (1.0 - ld2) / ld2_unc)
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)

        start,stop = ((-100. - veloffset) / veloffset_unc,
            (100. - veloffset) / veloffset_unc)
        p0[...,4] = truncnorm.rvs(start,stop
                ,loc=veloffset,scale=veloffset_unc,size=nwalkers)

        for i in xrange(self.nplanets):
            (T0,per,b,rprs,ecosw,
                esinw,rvamp) = self.fit_sol[i*7+5:i*7+12]
            b = 0.2
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*7+5] = np.random.normal(
                T0,T0_unc,size=nwalkers)
            p0[...,i*7+6] = np.random.normal(
                per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (0.5 - b) / b_unc
            p0[...,i*7+7] = truncnorm.rvs(
                start,stop
                ,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*7+8] = truncnorm.rvs(
                start,stop
                ,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (0.0 - ecosw) / ecosw_unc, (0.5 - ecosw) / ecosw_unc
            p0[...,i*7+9] = truncnorm.rvs(
                start,stop
                ,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (0.0 - esinw) / esinw_unc, (0.5 - esinw) / esinw_unc
            p0[...,i*7+10] = truncnorm.rvs(
                start,stop
                ,loc=esinw,scale=esinw_unc,size=nwalkers)
            p0[...,i*7+11] = np.random.normal(
                rvamp,rvamp_unc,size=nwalkers)

        #lcjitter
        start,stop = 0.0, 10.
        p0[...,-2] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*np.median(self.err),size=nwalkers)

        #rvjitter
        start,stop = 0.0, 10.
        p0[...,-1] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*np.median(self.rverr),size=nwalkers)
        return p0

    def cut_non_transit(self,ntdur=10):
        #make a mask for each planet candidate
        self.onlytransits = True
        tregion = np.zeros(self.nplanets)

        maskdat = np.zeros([self.npt,self.nplanets],dtype=bool)

        if self.n_ldparams == 2:
            addval = 0
        elif self.n_ldparams == 4:
            addval = 2
        for i in xrange(self.nplanets):
            T0 = self.fit_sol[i*7 + 5+addval]
            per = self.fit_sol[i*7 + 6+addval]
            rho = self.fit_sol[0]

            ars = self.get_ar(rho,per)
            tdur_dys = (1./ars) * per * (1./np.pi)

            #this is buggy because T0 is not nessessarily time of first transit
            #but time of a transit. So fudge.
            #subtract make T0 the first transit
            time0 = np.copy(T0)
            while True:
                if time0 - per < self.time[0]:
                    break
                else:
                    time0 = time0 - per

            ntransits = int((self.time[-1] - self.time[0]) / per) + 1
            t_times = np.arange(ntransits)*per + T0

            #make sure the first and last transit are not excluded even if
            #partially in the data
            t_times = np.r_[t_times,t_times[0] - per,t_times[-1] + per]

            for j in t_times:
                maskdat[:,i] = np.logical_or(maskdat[:,i],
                    np.logical_and(
                    self.time < j +tdur_dys*ntdur,
                    self.time > j - tdur_dys*ntdur) )
            tregion[i] = ntdur*tdur_dys
        #create a final mask that is the OR of the
        #individual masks
        finmask = np.zeros(self.npt)
        for i in xrange(self.nplanets):
            finmask = np.logical_or(finmask,maskdat[:,i])

        self.time = self.time[finmask]
        self.flux = self.flux[finmask]
        self.err = self.err[finmask]
        self._itime = self._itime[finmask]
        self._datatype = self._datatype[finmask]
        self.tregion = tregion


def logchi2_rv(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    rvtime,rvval,rverr,rvitime,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):

    """
    fitsol should have the format
    rho_0,zpt_0,ld1,ld2,veloffset
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0,rvamp_0

    fixed_sol should have
    dil,occ,ell,alb

    """

    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 100.:
        return minf

    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0

    rprs = fitsol[np.arange(nplanets)*7 + 8]
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = fitsol[np.arange(nplanets)*7 + 9]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*7 + 10]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*7 + 6]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf

    b = fitsol[np.arange(nplanets)*7 + 7]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*7 + 5]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf

    jitter_lc = fitsol[-2]
    if jitter_lc < 0.0:
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2

    jitter_rv = fitsol[-1]
    if jitter_rv < 0.0:
        return minf
    rverr_jit = np.sqrt(rverr**2 + jitter_rv**2)
    rverr_jit2 = rverr**2 + jitter_rv**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    time_model_calc = np.r_[time,rvtime]
    itime_model_calc = np.r_[itime,rvitime]
    datatype_model_calc  = np.r_[datatype,np.ones_like(rvtime)]

    model_lcrv = calc_model(fitsol_model_calc,nplanets,fixed_sol_model_calc,
            time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv[datatype_model_calc == 0] - 1.
    model_rv = model_lcrv[datatype_model_calc == 1]

    ### old likelihood
    # if rho_prior:
    #     rhoprior = (rho_0 - rho)**2 / rho_0_unc**2
    # else:
    #     rhoprior = 0.0

    # if ldp_prior:
    #     ldprior1 = (ld1_0 - ld1)*(ld1_0 - ld1) / ld1_0_unc**2
    #     ldprior2 = (ld2_0 - ld2)*(ld2_0 - ld2) / ld2_0_unc**2
    #     ldprior = ldprior1 + ldprior2
    # else:
    #     ldprior = 0.0

    # chi2prior = -0.5*(rhoprior+ldprior)

    ecc[ecc == 0.0] = 1.E-10
    #chi2ecc = np.log(ecc)

    # chi2lc = -0.5*np.sum(((model_lc - flux)* (model_lc - flux))
    #         / (err_jit*err_jit))

    # chi2rv = -0.5*np.sum(((model_rv - rvval) * (model_rv - rvval))
    #         / (rverr_jit*rverr_jit))

    # chi2const = -1.0*(np.sum(np.log(err_jit)) + np.sum(np.log(rverr_jit)))

    # chi2tot = chi2const + chi2lc + chi2rv + chi2prior

    # logp = chi2tot - np.sum(chi2ecc)

    ###
    ###new go at the log-likelihood
    ###

    # tpcon = 1. / (np.sqrt(2.*np.pi))

    # chi2lc = (
    #     - np.sum(0.5 * (model_lc - flux)**2 * (1./err_jit)**2)
    #     - (0.5 * npt_lc * np.log(2.*np.pi))
    #     - ((npt_lc + 1) * np.log(err_jit))
    #     )

    # chi2rv = (
    #     - np.sum(0.5 * (model_rv - rvval)**2 * (1./rverr_jit)**2)
    #     - (0.5 * npt_rv * np.log(2.*np.pi))
    #     - (npt_rv + 1) * np.sum(np.log(rverr_jit))
    #     )

    #chi2lc = np.sum(
    #    -np.log(tpcon * (1./err_jit)) +
    #        (model_lc - flux)**2 * (1./err_jit**2))

    #chi2rv = np.sum(
    #    -np.log(tpcon * (1./rverr_jit)) +
    #        (model_rv - rvval)**2 * (1./rverr_jit**2))

    #chi2ecc = np.sum(np.log(1./ecc))

    #if rho_prior:
    #    chi2rho = (-np.log(tpcon * (1./rho_0_unc)) +
    #                (rho_0 - rho)**2 * (1./rho_0_unc)**2)
    #else:
    #    chi2rho = 0.0

    #if ldp_prior:
    #    chi2ld1 = (-np.log(tpcon * (1./ld1_0_unc)) +
    #                (ld1_0 - ld1)**2 * (1./ld1_0_unc)**2)
    #
    #    chi2ld2 = (-np.log(tpcon * (1./ld2_0_unc)) +
    #                (ld2_0 - ld2)**2 * (1./ld2_0_unc)**2)
    #
    #    chi2ld = chi2ld1 + chi2ld2
    #else:
    #    chi2ld = 0.0

    # if rho_prior:
    #     chi2rho = (
    #         - 0.5 * (rho_0 - rho)**2 * (1./rho_0_unc)**2
    #         - (0.5 * np.log(2.*np.pi))
    #         - (2. * )
    #         )

    #chi2tot = -0.5*(chi2rv + chi2lc + chi2rho + chi2ld) + chi2ecc

    npt_lc = len(err_jit)
    npt_rv = len(rverr_jit)

    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )

    logrv = (
        - (npt_rv/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(rverr_jit2))
        - 0.5 * np.sum((model_rv - rvval)**2 / rverr_jit2)
        )

    if rho_prior:
        logrho = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(rho_0_unc)
            - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
            )
    else:
        rho_prior = 0.0

    if ldp_prior:
        logld1 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld1_0_unc)
            - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
            )

        logld2 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld2_0_unc)
            - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
            )

        logldp = logld1 + logld2
    else:
        logldp = 0.0

    logecc = - np.sum(np.log(ecc))

    logLtot = loglc + logrv + logrho + logldp + logecc

    return logLtot

def logchi2_rv_nontransitfudge(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    rvtime,rvval,rverr,rvitime,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):

    """
    this is a dirty fudge to get the second planet not to transit

    fitsol should have the format
    rho_0,zpt_0,ld1,ld2,veloffset
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0,rvamp_0

    fixed_sol should have
    dil,occ,ell,alb

    """

    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 100.:
        return minf

    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0


    rprs = fitsol[np.arange(nplanets)*7 + 8]
    b = fitsol[np.arange(nplanets)*7 + 7]

    ## fudgey part !!
    rprs[1] = 0.0
    b[1] = 0.0

    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = fitsol[np.arange(nplanets)*7 + 9]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*7 + 10]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*7 + 6]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf


    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*7 + 5]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf

    jitter_lc = fitsol[-2]
    if jitter_lc < 0.0:
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2

    jitter_rv = fitsol[-1]
    if jitter_rv < 0.0:
        return minf
    rverr_jit = np.sqrt(rverr**2 + jitter_rv**2)
    rverr_jit2 = rverr**2 + jitter_rv**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    time_model_calc = np.r_[time,rvtime]
    itime_model_calc = np.r_[itime,rvitime]
    datatype_model_calc  = np.r_[datatype,np.ones_like(rvtime)]

    model_lcrv = calc_model(fitsol_model_calc,nplanets,fixed_sol_model_calc,
            time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv[datatype_model_calc == 0] - 1.
    model_rv = model_lcrv[datatype_model_calc == 1]

    ecc[ecc == 0.0] = 1.E-10


    npt_lc = len(err_jit)
    npt_rv = len(rverr_jit)

    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )

    logrv = (
        - (npt_rv/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(rverr_jit2))
        - 0.5 * np.sum((model_rv - rvval)**2 / rverr_jit2)
        )

    if rho_prior:
        logrho = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(rho_0_unc)
            - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
            )
    else:
        rho_prior = 0.0

    if ldp_prior:
        logld1 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld1_0_unc)
            - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
            )

        logld2 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld2_0_unc)
            - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
            )

        logldp = logld1 + logld2
    else:
        logldp = 0.0

    logecc = - np.sum(np.log(ecc))

    logLtot = loglc + logrv + logrho + logldp + logecc

    return logLtot


def calc_model(fitsol,nplanets,fixed_sol,time,itime,ntt,tobs,omc,datatype):
    sol = np.zeros([8 + 11*nplanets])
    rho = fitsol[0]
    zpt = fitsol[1]
    ld1,ld2,ld3,ld4 = fixed_sol[0:4]
    dil = fixed_sol[4]
    veloffset = fitsol[2]

    fixed_stuff = fixed_sol[5:8]

    sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
        dil,veloffset,zpt])
    for i in xrange(nplanets):
        sol[8+(i*10):8+(i*10)+10] = np.r_[fitsol[3+i*7:10+i*7],fixed_stuff]

    tmodout = tmod.transitmodel(nplanets,sol,time,itime,
        ntt,tobs,omc,datatype)

    return tmodout

class transitemcee_rv_hetero(transitemcee_rv,
                             transitemcee.transitemcee_fitldp,
                             transitemcee.transitemcee):

    def __init__(self,nplanets,cadence=1625.3,rvcadence=1800.0,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):
        super(transitemcee_rv_hetero,self).__init__(
            nplanets,cadence=cadence,rvcadence=rvcadence,
            ldfileloc=ldfileloc,
            codedir=codedir)
        sys.path.append(codedir)


    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw,rvamp
        """

        rho_unc = 0.001
        zpt_unc = 1.E-8
        ld1_unc = 0.05
        ld2_unc = 0.05
        veloffset_unc = 100.
        T0_unc = 0.0002
        per_unc = 0.00005
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        rvamp_unc = 10.

        if self.n_ldparams == 2:
            p0 = np.zeros([nwalkers,
                5+self.nplanets*7+1+len(self.rverr)])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]
        veloffset = self.fit_sol[4]

        start,stop = ((0.0001 - rho) / rho_unc,
            (30.0 - rho) / rho_unc)
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,
            size=nwalkers)

        start,stop = ((0.0 - ld1) / ld1_unc,
            (1.0 - ld1) / ld1_unc)
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = ((0.0 - ld2) / ld2_unc,
            (1.0 - ld2) / ld2_unc)
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)

        start,stop = ((-100. - veloffset) / veloffset_unc,
            (100. - veloffset) / veloffset_unc)
        p0[...,4] = truncnorm.rvs(start,stop
                ,loc=veloffset,scale=veloffset_unc,size=nwalkers)

        for i in xrange(self.nplanets):
            (T0,per,b,rprs,ecosw,
                esinw,rvamp) = self.fit_sol[i*7+5:i*7+12]
            b = 0.2
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*7+5] = np.random.normal(
                T0,T0_unc,size=nwalkers)
            p0[...,i*7+6] = np.random.normal(
                per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (0.5 - b) / b_unc
            p0[...,i*7+7] = truncnorm.rvs(
                start,stop
                ,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*7+8] = truncnorm.rvs(
                start,stop
                ,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (0.0 - ecosw) / ecosw_unc, (0.5 - ecosw) / ecosw_unc
            p0[...,i*7+9] = truncnorm.rvs(
                start,stop
                ,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (0.0 - esinw) / esinw_unc, (0.5 - esinw) / esinw_unc
            p0[...,i*7+10] = truncnorm.rvs(
                start,stop
                ,loc=esinw,scale=esinw_unc,size=nwalkers)
            p0[...,i*7+11] = np.random.normal(
                rvamp,rvamp_unc,size=nwalkers)

        #lcjitter
        start,stop = 0.0, 10.
        p0[...,5+self.nplanets*7] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*np.median(self.err),size=nwalkers)

        #rvjitter
        start,stop = 0.0, 10.
        p0[...,5+self.nplanets*7+1:] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*np.median(self.rverr),
            size=[nwalkers,len(self.rverr)])
        return p0

def logchi2_rv_hetero(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    rvtime,rvval,rverr,rvitime,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):

    """
    fitsol should have the format
    rho_0,zpt_0,ld1,ld2,veloffset
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0,rvamp_0

    fixed_sol should have
    dil,occ,ell,alb

    """

    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 100.:
        return minf

    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0

    rprs = fitsol[np.arange(nplanets)*7 + 8]
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = fitsol[np.arange(nplanets)*7 + 9]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*7 + 10]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*7 + 6]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf

    b = fitsol[np.arange(nplanets)*7 + 7]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*7 + 5]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf

    jitter_lc = fitsol[(nplanets-1)*7 + 12]
    if jitter_lc < 0.0:
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2

    jitter_rv = fitsol[(nplanets-1)*7 + 13:]
    if np.any(jitter_rv < 0.0):
        return minf
    rverr_jit = np.sqrt(rverr**2 + jitter_rv**2)
    rverr_jit2 = rverr**2 + jitter_rv**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    time_model_calc = np.r_[time,rvtime]
    itime_model_calc = np.r_[itime,rvitime]
    datatype_model_calc  = np.r_[datatype,np.ones_like(rvtime)]

    model_lcrv = calc_model(fitsol_model_calc,nplanets,fixed_sol_model_calc,
            time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv[datatype_model_calc == 0] - 1.
    model_rv = model_lcrv[datatype_model_calc == 1]

    ### old likelihood
    # if rho_prior:
    #     rhoprior = (rho_0 - rho)**2 / rho_0_unc**2
    # else:
    #     rhoprior = 0.0

    # if ldp_prior:
    #     ldprior1 = (ld1_0 - ld1)*(ld1_0 - ld1) / ld1_0_unc**2
    #     ldprior2 = (ld2_0 - ld2)*(ld2_0 - ld2) / ld2_0_unc**2
    #     ldprior = ldprior1 + ldprior2
    # else:
    #     ldprior = 0.0

    # chi2prior = -0.5*(rhoprior+ldprior)

    ecc[ecc == 0.0] = 1.E-10
    #chi2ecc = np.log(ecc)

    # chi2lc = -0.5*np.sum(((model_lc - flux)* (model_lc - flux))
    #         / (err_jit*err_jit))

    # chi2rv = -0.5*np.sum(((model_rv - rvval) * (model_rv - rvval))
    #         / (rverr_jit*rverr_jit))

    # chi2const = -1.0*(np.sum(np.log(err_jit)) + np.sum(np.log(rverr_jit)))

    # chi2tot = chi2const + chi2lc + chi2rv + chi2prior

    # logp = chi2tot - np.sum(chi2ecc)

    ###
    ###new go at the log-likelihood
    ###
    # tpcon = 1. / (np.sqrt(2.*np.pi))

    # chi2lc = np.sum(
    #     -np.log(tpcon * (1./err_jit)) +
    #         (model_lc - flux)**2 * (1./err_jit**2))

    # chi2rv = np.sum(
    #     -np.log(tpcon * (1./rverr_jit)) +
    #         (model_rv - rvval)**2 * (1./rverr_jit**2))

    # chi2ecc = np.sum(np.log(1./ecc))

    # if rho_prior:
    #     chi2rho = (-np.log(tpcon * (1./rho_0_unc)) +
    #                 (rho_0 - rho)**2 * (1./rho_0_unc)**2)
    # else:
    #     chi2rho = 0.0

    # if ldp_prior:
    #     chi2ld1 = (-np.log(tpcon * (1./ld1_0_unc)) +
    #                 (ld1_0 - ld1)**2 * (1./ld1_0_unc)**2)

    #     chi2ld2 = (-np.log(tpcon * (1./ld2_0_unc)) +
    #                 (ld2_0 - ld2)**2 * (1./ld2_0_unc)**2)

    #     chi2ld = chi2ld1 + chi2ld2
    # else:
    #     chi2ld = 0.0

    # chi2tot = -0.5*(chi2rv + chi2lc + chi2rho + chi2ld) + chi2ecc

    # return chi2tot

    npt_lc = len(err_jit)
    npt_rv = len(rverr_jit)

    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )

    logrv = (
        - (npt_rv/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(rverr_jit2))
        - 0.5 * np.sum((model_rv - rvval)**2 / rverr_jit2)
        )

    if rho_prior:
        logrho = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(rho_0_unc**2)
            - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
            )
    else:
        rho_prior = 0.0

    if ldp_prior:
        logld1 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld1_0_unc**2)
            - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
            )

        logld2 = (
            - 0.5 * np.log(2.*np.pi)
            - 0.5 * np.log(ld2_0_unc**2)
            - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
            )

        logldp = logld1 + logld2
    else:
        logldp = 0.0

    logecc = - np.sum(np.log(ecc))

    logLtot = loglc + logrv + logrho + logldp + logecc

    return logLtot
