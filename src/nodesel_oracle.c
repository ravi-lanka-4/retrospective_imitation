/**@file   nodesel_dagger.c
 * @brief  uct node selector which balances exploration and exploitation by considering node visits
 * @author Gregor Hendel
 *
 * the UCT node selection rule selects the next leaf according to a mixed score of the node's actual lower bound
 *
 * The idea of UCT node selection for MIP appeared in:
 *
 * The authors adapted a game-tree exploration scheme called UCB to MIP trees. Starting from the root node as current node,
 *
 * The node selector features several parameters:
 *
 * @note It should be avoided to switch to uct node selection after the branch and bound process has begun because
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include <assert.h>
#include <string.h>
#include "nodesel_oracle.h"
#include "helper.h"
#include "feat.h"
#include "policy.h"
#include "helper.h"
#include "struct_policy.h"
#include "scip/def.h"
#include "scip/sol.h"
#include "scip/tree.h"
#include "scip/stat.h"
#include "scip/struct_set.h"
#include "scip/struct_scip.h"
#include "scip/struct_var.h"
#include "scip/pub_tree.h"

#define NODESEL_NAME            "oracle"
#define NODESEL_DESC            "node selector which selects node according to a optimal solution"
#define NODESEL_STDPRIORITY     10
#define NODESEL_MEMSAVEPRIORITY 0

#define DEFAULT_FILENAME        ""

static
SCIP_RETCODE createFeatDiff(
   SCIP*                scip,
   SCIP_NODE*           node,
   SCIP_NODESELDATA*    nodeseldata
);

/*
 * Data structures
 */

/** node selector data */
struct SCIP_NodeselData
{
   char*              solfname;           /**< name of the solution file */
   char*              polfname;           /**< name of the solution file */
   char*              trjfname;           /**< name of the trajectory file */
   FILE*              wfile;
   FILE*              trjfile1;           /** select is a ranking function so write them separately */
   FILE*              trjfile2;
   SCIP_FEAT*         feat;
   SCIP_Longint       optnodenumber;      /**< successively assigned number of the node */
   SCIP_Bool          negate;
   int                nerrors;            /**< number of wrong ranking of a pair of nodes */
   int                ncomps;             /**< total number of comparisons */
   SCIP_Real          scale;
   SCIP_Real          margin;
   SCIP_Real          depth_threshold;

   /* Optimal traces */
   SCIP_Real          ogapThreshold;      /**< Treshold to consider the solutions */
   SCIP_Longint       nfeasiblesols;
   SCIP_SOL**         feasiblesols;             /**< optimal solution */
   SCIP_FEAT**        feasiblefeat;
   SCIP_Bool*         solflag;

   /* Problem related features */
   char*              probfeatsfname;
   SCIP_Real*         probfeats;
   int                probfeatsize;
};

/*
 * Local methods
 */

/** check if the given node include the optimal solution */
/* TODO: remove to ischecked */
SCIP_Bool SCIPnodeCheckOptimal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< the node in question */
   SCIP_SOL*             optsol,             /**< node selector data */
   int                   idx,
   SCIP_Bool             useSol,
   SCIP_Real             margin
   )
{
   /* get parent branch vars lead to this node */

   SCIP_VAR**            branchvars;         /* array of variables on which the branchings has been performed in all ancestors */
   SCIP_Real*            branchbounds;       /* array of bounds which the branchings in all ancestors set */
   SCIP_BOUNDTYPE*       boundtypes;         /* array of boundtypes which the branchings in all ancestors set */
   int                   nbranchvars;        /* number of variables on which branchings have been performed in all ancestors
                                              *   if this is larger than the array size, arrays should be reallocated and method should be called again */
   int                   branchvarssize;     /* available slots in arrays */ 

   int i;
   SCIP_NODE* parent;
   SCIP_Bool isoptimal = TRUE;

   assert(optsol != NULL);
   assert(node != NULL);

   SCIPdebugMessage("checking node %d\n", (int)SCIPnodeGetNumber(node));

   assert(SCIPnodeIsOptchecked(node) == FALSE);

   /* don't consider root node */
   assert(SCIPnodeGetDepth(node) != 0);

   /* check parent: if parent is not optimal, its subtree is not optimal */
   parent = SCIPnodeGetParent(node);

   if (parent == NULL){
      SCIPnodeSetSolFlag(node, idx, TRUE);
      return TRUE;
   }

   /* root is always optimal */
   branchvarssize = 1;

   /* memory allocation */
   SCIP_CALL( SCIPallocBufferArray(scip, &branchvars, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &branchbounds, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &boundtypes, branchvarssize) );

   SCIPnodeGetParentBranchings(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize);

   /* if the arrays were to small, we have to reallocate them and recall SCIPnodeGetParentBranchings */
   if( nbranchvars > branchvarssize )
   {
      branchvarssize = nbranchvars;

      /* memory reallocation */
      SCIP_CALL( SCIPreallocBufferArray(scip, &branchvars, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &branchbounds, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &boundtypes, branchvarssize) );

      SCIPnodeGetParentBranchings(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize);
      assert(nbranchvars == branchvarssize);
   }

   /* check optimality */
   assert(nbranchvars >= 1);
   for( i = 0; i < nbranchvars; ++i)
   {
      SCIP_Real optval = SCIPgetSolVal(scip, optsol, branchvars[i]);

      /* Keep the features */
      int step=-1, dim=-1, side=-1;
      SCIPnodeSetBranchBound(node, branchbounds[i]);
      SCIPnodeSetBoundType(node, boundtypes[i]);

      if (SCIPgetNBinVars(scip) != 0){
        if (SCIPgetNContVars(scip) != 0){
          /* pSulu with binary variables */
          sscanf(SCIPvarGetName(branchvars[i]), "t_z_%d_%d_%d", &step, &dim, &side);
          SCIPnodeSetIndex(node, step, 0);
          SCIPnodeSetIndex(node, dim, 1);
          SCIPnodeSetIndex(node, side, 1);
          assert(step != -1);
          assert(dim != -1);
          assert(side != -1);
        }
        else{
          /* Only binary variables no continuous variables -- mvc */
          sscanf(SCIPvarGetName(branchvars[i]), "t_v%d", &step);
          SCIPnodeSetIndex(node, step, 0);
          assert(step != -1);
        }
      }
      else{
        /* Only continuous variables -- pSulu spatial */
        sscanf(SCIPvarGetName(branchvars[i]), "t_x_%d_%d", &step, &dim);
        SCIPnodeSetIndex(node, step, 0);
        SCIPnodeSetIndex(node, dim, 1);
        assert(step != -1);
        assert(dim != -1);
      }

      if (optval == -1){
         isoptimal = MY_UNKNOWN;
         break;
      }

      if( (boundtypes[i] == SCIP_BOUNDTYPE_LOWER && optval < branchbounds[i] - margin) ||
          (boundtypes[i] == SCIP_BOUNDTYPE_UPPER && optval > branchbounds[i] + margin) )
      {
         isoptimal = FALSE;
         break;
      }
      if( (boundtypes[i] == SCIP_BOUNDTYPE_LOWER && optval < branchbounds[i]) ||
          (boundtypes[i] == SCIP_BOUNDTYPE_UPPER && optval > branchbounds[i]) )
      {
         isoptimal = MY_UNKNOWN;
         break;
      }
   }

   /* free all local memory */
   SCIPfreeBufferArray(scip, &branchvars);
   SCIPfreeBufferArray(scip, &boundtypes);
   SCIPfreeBufferArray(scip, &branchbounds);

   /* Set the flag for the solution */
   if (!useSol)
   {
     if( (SCIPnodeGetDepth(parent) > 0)  && !SCIPnodeGetSolFlag(parent, idx)){
        SCIPnodeSetSolFlag(node, idx, FALSE);
        return FALSE;
     }
   }
 
   SCIPnodeSetSolFlag(node, idx, isoptimal);
   return isoptimal;
}

/** read the optimal solution (modified from readSol in reader_sol.c -- don't connect the solution with primal solutions) */
/** TODO: currently the read objective is wrong, since it doesn not include objetives from multi-aggregated variables */
SCIP_RETCODE SCIPreadOptSol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           fname,              /**< name of the input file */
   SCIP_SOL**            sol                 /**< pointer to store the solution */
   )
{
   SCIP_FILE* file;
   SCIP_Bool error;
   SCIP_Bool unknownvariablemessage;
   SCIP_Bool usevartable;
   int lineno;

   assert(scip != NULL);
   assert(fname != NULL);

   SCIP_CALL( SCIPgetBoolParam(scip, "misc/usevartable", &usevartable) );

   if( !usevartable )
   {
      SCIPerrorMessage("Cannot read solution file if vartable is disabled. Make sure parameter 'misc/usevartable' is set to TRUE.\n");
      return SCIP_READERROR;
   }

   /* open input file */
   file = SCIPfopen(fname, "r");
   if( file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", fname);
      SCIPprintSysError(fname);
      return SCIP_NOFILE;
   }

   /* create zero solution */
   SCIP_CALL( SCIPcreateSolSelf(scip, sol, NULL) );
   assert(SCIPsolIsOriginal(*sol) == TRUE);

   /* read the file */
   error = FALSE;
   unknownvariablemessage = FALSE;
   lineno = 0;
   while( !SCIPfeof(file) && !error )
   {
      char buffer[SCIP_MAXSTRLEN];
      char varname[SCIP_MAXSTRLEN];
      char valuestring[SCIP_MAXSTRLEN];
      char objstring[SCIP_MAXSTRLEN];
      SCIP_VAR* var;
      SCIP_Real value;
      int nread;

      /* get next line */
      if( SCIPfgets(buffer, (int) sizeof(buffer), file) == NULL )
         break;
      lineno++;

      /* there are some lines which may preceed the solution information */
      if( strncasecmp(buffer, "solution status:", 16) == 0 || strncasecmp(buffer, "#Objective value", 16) == 0 ||
         strncasecmp(buffer, "Log started", 11) == 0 || strncasecmp(buffer, "Variable Name", 13) == 0 ||
         strncasecmp(buffer, "All other variables", 19) == 0 || strncasecmp(buffer, "\n", 1) == 0 ||
         strncasecmp(buffer, "NAME", 4) == 0 || strncasecmp(buffer, "ENDATA", 6) == 0 )    /* allow parsing of SOL-format on the MIPLIB 2003 pages */
         continue;

      /* parse the line */
      nread = sscanf(buffer, "%s %s %s\n", varname, valuestring, objstring);
      if( nread < 2 )
      {
         SCIPerrorMessage("Invalid input line %d in solution file <%s>: <%s>.\n", lineno, fname, buffer);
         error = TRUE;
         break;
      }

      /* find the variable */
      var = SCIPfindVar(scip, varname);
      if( var == NULL )
      {
         if( !unknownvariablemessage )
         {
            SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "unknown variable <%s> in line %d of solution file <%s>\n",
               varname, lineno, fname);
            SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "  (further unknown variables are ignored)\n");
            unknownvariablemessage = TRUE;
         }
         continue;
      }

      /* cast the value */
      if( strncasecmp(valuestring, "inv", 3) == 0 )
         continue;
      else if( strncasecmp(valuestring, "+inf", 4) == 0 || strncasecmp(valuestring, "inf", 3) == 0 )
         value = SCIPinfinity(scip);
      else if( strncasecmp(valuestring, "-inf", 4) == 0 )
         value = -SCIPinfinity(scip);
      else
      {
         nread = sscanf(valuestring, "%lf", &value);
         if( nread != 1 )
         {
            SCIPerrorMessage("Invalid solution value <%s> for variable <%s> in line %d of solution file <%s>.\n",
               valuestring, varname, lineno, fname);
            error = TRUE;
            break;
         }
      }

      /* set the solution value of the variable, if not multiaggregated */
      if( SCIPisTransformed(scip) && SCIPvarGetStatus(SCIPvarGetProbvar(var)) == SCIP_VARSTATUS_MULTAGGR )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "ignored solution value for multiaggregated variable <%s>\n", SCIPvarGetName(var));
      }
      else
      {
         SCIP_RETCODE retcode;
         retcode =  SCIPsetSolVal(scip, *sol, var, value);

         if( retcode == SCIP_INVALIDDATA )
         {
            if( SCIPvarGetStatus(SCIPvarGetProbvar(var)) == SCIP_VARSTATUS_FIXED )
            {
               SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "ignored conflicting solution value for fixed variable <%s>\n",
                  SCIPvarGetName(var));
            }
            else
            {
               SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "ignored solution value for multiaggregated variable <%s>\n",
                  SCIPvarGetName(var));
            }
         }
         else
         {
            SCIP_CALL( retcode );
         }
      }
   }

   /* close input file */
   SCIPfclose(file);

   if( !error )
   {
      /* display result */
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "optimal solution from solution file <%s> was %s\n",
         fname, "read");
      return SCIP_OKAY;
   }
   else
   {
      /* free solution */
      SCIP_CALL( SCIPfreeSolSelf(scip, sol) );

      return SCIP_READERROR;
   }
}

/*
 * Callback methods of node selector
 */

/** solving process initialization method of node selector (called when branch and bound process is about to begin) */
static
SCIP_DECL_NODESELINIT(nodeselInitOracle)
{
   SCIP_NODESELDATA* nodeseldata;
   SCIP_Real bestobj;
   int i=0;
   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

//    printf("#############################################################################\n");
//    printf("Switching optimal nodes to explore sub optimal ones first; to a certain depth\n");
//    printf("#############################################################################\n");

   /* Read problem specific features */
   nodeseldata->probfeats = NULL;
   if(strcmp(nodeseldata->probfeatsfname, DEFAULT_FILENAME) != 0)
   {
      FILE* featsfp= fopen(nodeseldata->probfeatsfname, "r");

      /* Assumes that the first line is the size of the feats */
      fscanf(featsfp, "%d\n", &(nodeseldata->probfeatsize));
      SCIP_ALLOC( BMSallocMemoryArray(&(nodeseldata->probfeats), nodeseldata->probfeatsize ));

      /* Read the features */
      readProbFeats(featsfp, nodeseldata->probfeats, nodeseldata->probfeatsize);
      fclose(featsfp);
   }
   else{
      nodeseldata->probfeatsize = 0;
   }

   /* solfname should be set before including nodeprudagger */
   assert(nodeseldata->solfname != NULL);
   nodeseldata->nfeasiblesols = 1;

   // Assign memory for the sols
   BMSallocMemoryArray(&(nodeseldata->feasiblesols), nodeseldata->nfeasiblesols);
   nodeseldata->feasiblesols[0] = NULL;
   SCIP_CALL( SCIPreadOptSol(scip, nodeseldata->solfname, &(nodeseldata->feasiblesols[0])));
   assert(nodeseldata->feasiblesols[0] != NULL);

   // Get best objective value from all the solutions 
   bestobj = getBestObj(nodeseldata->feasiblesols, nodeseldata->nfeasiblesols);

   /* Create the flag for using the solution based on the threshold */
   BMSallocMemoryArray(&(nodeseldata->solflag), nodeseldata->nfeasiblesols);
   for(i=0; i<nodeseldata->nfeasiblesols; i++){
      SCIP_Real ogap;
      SCIP_Real cobj = SCIPsolGetOrigObj(nodeseldata->feasiblesols[i]);
      ogap = fabs((cobj - bestobj)/(0.01+bestobj));
      if (ogap <= nodeseldata->ogapThreshold)
         nodeseldata->solflag[i] = TRUE;
      else
         nodeseldata->solflag[i] = FALSE;
   }

   /* open trajectory file for writing */
   /* open in appending mode for writing training file from multiple problems */
   nodeseldata->trjfile1 = NULL;
   nodeseldata->trjfile2 = NULL;
   if( nodeseldata->trjfname != NULL )
   {
      char wfname[SCIP_MAXSTRLEN];
      strcpy(wfname, nodeseldata->trjfname);
      strcat(wfname, ".weight");
      nodeseldata->wfile = fopen(wfname, "a");

#ifdef LIBLINEAR
      nodeseldata->trjfile1 = fopen(nodeseldata->trjfname, "a");
#else
      strcpy(wfname, nodeseldata->trjfname);
      strcat(wfname, ".1");
      nodeseldata->trjfile1 = fopen(wfname, "a");

      strcpy(wfname, nodeseldata->trjfname);
      strcat(wfname, ".2");
      nodeseldata->trjfile2 = fopen(wfname, "a");
#endif
   }

   /* create feat */
   nodeseldata->feat = NULL;
   SCIP_CALL( SCIPfeatCreate(scip, &nodeseldata->feat, NULL, 
                              SCIP_FEATNODESEL_SIZE + nodeseldata->probfeatsize + BUFF, 
                              SCIP_FEATNODESEL_SIZE) );
   assert(nodeseldata->feat != NULL);
   SCIPfeatSetMaxDepth(nodeseldata->feat, SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));

   /* create optimal node feat */
   nodeseldata->feasiblefeat = NULL;
   BMSallocMemoryArray(&(nodeseldata->feasiblefeat), nodeseldata->nfeasiblesols);
   for(i=0; i<nodeseldata->nfeasiblesols; i++){
      SCIP_CALL( SCIPfeatCreate(scip, &(nodeseldata->feasiblefeat[i]), NULL, 
                                 SCIP_FEATNODESEL_SIZE + nodeseldata->probfeatsize + BUFF, 
                                 SCIP_FEATNODESEL_SIZE) );
      assert(&(nodeseldata->feasiblefeat[i]) != NULL);
      SCIPfeatSetMaxDepth(nodeseldata->feasiblefeat[i], SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip));
   }

   nodeseldata->optnodenumber = -1;
   nodeseldata->negate = TRUE;

   nodeseldata->nerrors = 0;
   nodeseldata->ncomps = 0;

   /* fclose(normF); */
   return SCIP_OKAY;
}

/** deinitialization method of node selector (called before transformed problem is freed) */
static
SCIP_DECL_NODESELEXIT(nodeselExitOracle)
{
   int i=0;
   SCIP_NODESELDATA* nodeseldata;
   assert(scip != NULL);
   assert(nodesel != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);

   assert(nodeseldata->feasiblesols != NULL);
   for(i=0; i<nodeseldata->nfeasiblesols; i++){
      assert(nodeseldata->feasiblesols[i] != NULL);
      SCIP_CALL( SCIPfreeSolSelf(scip, &(nodeseldata->feasiblesols[i])) );
      SCIP_CALL( SCIPfeatFree(scip, &(nodeseldata->feasiblefeat[i])) );;
   }
   BMSfreeMemory(&(nodeseldata->feasiblesols));
   BMSfreeMemory(&(nodeseldata->feasiblefeat));
   BMSfreeMemory(&(nodeseldata->solflag));

   if( nodeseldata->trjfile1 != NULL)
   {
      fclose(nodeseldata->wfile);
      fclose(nodeseldata->trjfile1);
#ifndef LIBLINEAR
      fclose(nodeseldata->trjfile2);
#endif
   }

   if(nodeseldata->probfeats != NULL)
      BMSfreeMemory(&(nodeseldata->probfeats));

   assert(nodeseldata->feat != NULL);
   SCIP_CALL( SCIPfeatFree(scip, &nodeseldata->feat) );

   return SCIP_OKAY;
}

/** destructor of node selector to free user data (called when SCIP is exiting) */
static
SCIP_DECL_NODESELFREE(nodeselFreeOracle)
{
   SCIP_NODESELDATA* nodeseldata;
   nodeseldata = SCIPnodeselGetData(nodesel);

   assert(nodeseldata != NULL);

   SCIPfreeBlockMemory(scip, &nodeseldata);

   SCIPnodeselSetData(nodesel, NULL);

   return SCIP_OKAY;
}

/** node selection method of node selector */
static
SCIP_DECL_NODESELSELECT(nodeselSelectOracle)
{
   SCIP_NODESELDATA* nodeseldata;
   SCIP_NODE** leaves;
   SCIP_NODE** children;
   SCIP_NODE** siblings;
   int nleaves;
   int nsiblings;
   int nchildren;
   SCIP_Bool optchild;
   int i;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(scip != NULL);
   assert(selnode != NULL);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   /* collect leaves, children and siblings data */
   SCIP_CALL( SCIPgetOpenNodesData(scip, &leaves, &children, &siblings, &nleaves, &nchildren, &nsiblings) );

   /* check newly created nodes */
   optchild = FALSE;
   for( i = 0; i < nchildren; i++)
   {
      // Check if memory is allocated for this node; if not, allocate
      SCIPnodeAllocSolFlag(children[i], nodeseldata->nfeasiblesols);

      /* check optimality */
      assert( ! SCIPnodeIsOptchecked(children[i]) );

      {
         SCIP_Longint solIdx;
         solIdx = getSolIndex(scip, children[i], nodeseldata->feasiblesols, 
                                    nodeseldata->solflag, nodeseldata->nfeasiblesols, nodeseldata->margin);

         /* populate feats */
         SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feat, \
                                    nodeseldata->probfeats, nodeseldata->probfeatsize);


         /* Get feasible node feats and update index */
         if(solIdx != -1){
            SCIPcalcNodeselFeat(scip, children[i], nodeseldata->feasiblefeat[solIdx], \
                                    nodeseldata->probfeats, nodeseldata->probfeatsize);
            SCIPnodeSetSolIdx(children[i], solIdx);
         }

         SCIPnodeSetOptchecked(children[i]);
      }

      if( SCIPnodeIsOptimal(children[i]) )
      {
#ifndef NDEBUG
         SCIPdebugMessage("opt node #%"SCIP_LONGINT_FORMAT"\n", SCIPnodeGetNumber(children[i]));
#endif
         nodeseldata->optnodenumber = SCIPnodeGetNumber(children[i]);
         optchild = TRUE;
      }
   }

   /* write examples */
   if( nodeseldata->trjfile1 != NULL )
   {
     for( i = 0; i < nchildren; i++){
            if(SCIPnodeIsOptimal(children[i]) == MY_UNKNOWN)
                  continue;

            createFeatDiff(scip, children[i], nodeseldata);
     }
     if(optchild)
     {
         /* Complete pair-wise feature created only if there is an 
            optimal solution among new children */
         for( i = 0; i < nsiblings; i++ ){
            if(SCIPnodeIsOptimal(siblings[i]) == MY_UNKNOWN)
                  continue;

            createFeatDiff(scip, siblings[i], nodeseldata);
         }
         for( i = 0; i < nleaves; i++ ){
            if(SCIPnodeIsOptimal(leaves[i]) == MY_UNKNOWN)
                  continue;

            createFeatDiff(scip, leaves[i], nodeseldata);
         }
      }
   }

   *selnode = SCIPgetBestNode(scip);

   return SCIP_OKAY;
}

/** node comparison method of oracle node selector */
static
SCIP_DECL_NODESELCOMP(nodeselCompOracle)
{  /*lint --e{715}*/
   SCIP_Bool isoptimal1;
   SCIP_Bool isoptimal2;

   assert(nodesel != NULL);
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(scip != NULL);

   assert(SCIPnodeIsOptchecked(node1) == TRUE);
   assert(SCIPnodeIsOptchecked(node2) == TRUE);

   isoptimal1 = SCIPnodeIsOptimal(node1);
   isoptimal2 = SCIPnodeIsOptimal(node2);

//    /* Always expand sub optimal node -- and let pruner take care */
//    {
//       SCIP_Bool tmp = isoptimal1;
//       isoptimal1 = isoptimal2;
//       isoptimal2 = tmp;
//    }

   if( isoptimal1 == TRUE )
      return -1;
   else if( isoptimal2 == TRUE )
      return +1;
   else
   {
      int depth1;
      int depth2;

      depth1 = SCIPnodeGetDepth(node1);
      depth2 = SCIPnodeGetDepth(node2);
      if( depth1 > depth2 )
         return -1;
      else if( depth1 < depth2 )
         return +1;
      else
      {
         SCIP_Real lowerbound1;
         SCIP_Real lowerbound2;

         lowerbound1 = SCIPnodeGetLowerbound(node1);
         lowerbound2 = SCIPnodeGetLowerbound(node2);
         if( SCIPisLT(scip, lowerbound1, lowerbound2) )
            return -1;
         else if( SCIPisGT(scip, lowerbound1, lowerbound2) )
            return +1;
         if( lowerbound1 < lowerbound2 )
            return -1;
         else if( lowerbound1 > lowerbound2 )
            return +1;
         else
            return 0;
      }
   }
}

static
SCIP_RETCODE createFeatDiff(
   SCIP*                scip,
   SCIP_NODE*           node,
   SCIP_NODESELDATA*    nodeseldata
)
{
   int csolidx = SCIPnodeGetSolIdx(node);
   int cnum    = SCIPnodeGetNumber(node);
   int j = 0;

   SCIPcalcNodeselFeat(scip, node, nodeseldata->feat, \
                                    nodeseldata->probfeats, nodeseldata->probfeatsize);

   for(j=0; j<nodeseldata->nfeasiblesols; j++)
   {

      int feasnum = SCIPfeatGetNumber(nodeseldata->feasiblefeat[j]);

      if (llabs(SCIPnodeGetDepth(node) - SCIPfeatGetDepth(nodeseldata->feasiblefeat[j])) > nodeseldata->depth_threshold){
         /* Gap too much -- ignore */
         continue;
      }

      if (!(nodeseldata->solflag[j])){
         continue;
      }
      else if ((feasnum == -1) || (feasnum == cnum)){
         // No node belonging to this solution or child belongs to the same sol
         continue;
      }
      else{
         nodeseldata->negate ^= 1;
         if (csolidx == -1)
            /* condition when the node does not belong to an optimal path */
            SCIPfeatDiffLIBSVMPrint(scip, nodeseldata->trjfile1, nodeseldata->trjfile2,\
                                          nodeseldata->wfile, nodeseldata->feasiblefeat[j], \
                                          nodeseldata->feat, 1, nodeseldata->scale, nodeseldata->negate);
         else{
            /* optimal path */
            SCIP_Real cobj = SCIPsolGetOrigObj(nodeseldata->feasiblesols[csolidx]);
            SCIP_Real otherobj = SCIPsolGetOrigObj(nodeseldata->feasiblesols[j]);
            SCIP_Bool label = (cobj > otherobj)? 1 : -1;
            SCIPfeatDiffLIBSVMPrint(scip, nodeseldata->trjfile1, nodeseldata->trjfile2, \
                                          nodeseldata->wfile, nodeseldata->feasiblefeat[j], \
                                          nodeseldata->feat, label, nodeseldata->scale, nodeseldata->negate);
         }
      }
   }
   return SCIP_OKAY;
}

/*
 * node selector specific interface methods
 */

/** creates the uct node selector and includes it in SCIP */
SCIP_RETCODE SCIPincludeNodeselOracle(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODESELDATA* nodeseldata;
   SCIP_NODESEL* nodesel;

   /* create dagger node selector data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &nodeseldata) );

   nodesel = NULL;
   nodeseldata->feasiblesols = NULL;
   nodeseldata->solfname = NULL;
   nodeseldata->trjfname = NULL;
   nodeseldata->probfeatsfname = NULL;
   nodeseldata->scale = 1;

   /* use SCIPincludeNodeselBasic() plus setter functions if you want to set callbacks one-by-one and your code should
    * compile independent of new callbacks being added in future SCIP versions
    */
   SCIP_CALL( SCIPincludeNodeselBasic(scip, &nodesel, NODESEL_NAME, NODESEL_DESC, NODESEL_STDPRIORITY,
          NODESEL_MEMSAVEPRIORITY, nodeselSelectOracle, nodeselCompOracle, nodeseldata) );

   assert(nodesel != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetNodeselCopy(scip, nodesel, NULL) );
   SCIP_CALL( SCIPsetNodeselInit(scip, nodesel, nodeselInitOracle) );
   SCIP_CALL( SCIPsetNodeselExit(scip, nodesel, nodeselExitOracle) );
   SCIP_CALL( SCIPsetNodeselFree(scip, nodesel, nodeselFreeOracle) );

   /* add dagger node selector parameters */
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/solfname",
         "name of the optimal solution file",
         &nodeseldata->solfname, FALSE, DEFAULT_FILENAME, NULL, NULL) );
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/trjfname",
         "name of the file to write node selection trajectories",
         &nodeseldata->trjfname, FALSE, DEFAULT_FILENAME, NULL, NULL) );
   SCIP_CALL( SCIPaddStringParam(scip,
         "nodeselection/"NODESEL_NAME"/probfeatsfname",
         "name of the file with prob specific features",
         &nodeseldata->probfeatsfname, FALSE, DEFAULT_FILENAME, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip, 
         "nodeselection/"NODESEL_NAME"/scale",
         "Scale the weights of the data points", 
         &nodeseldata->scale, FALSE, 1, 0.001, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip, 
         "nodeselection/"NODESEL_NAME"/margin",
         "Scale the weights of the data points", 
         &nodeseldata->margin, FALSE, 0, 0, 0.05, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip, 
         "nodeselection/"NODESEL_NAME"/depth_treshold",
         "depth threshold for accumulating data", 
         &nodeseldata->depth_threshold, FALSE, 5, 0, SCIP_REAL_MAX, NULL, NULL) );


   return SCIP_OKAY;
}
