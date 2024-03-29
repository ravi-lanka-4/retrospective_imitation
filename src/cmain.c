/**@file   cmain.c
 * @brief  Main file for training/testing node selector/pruner using dagger 
 * @author He He 
 */

#include <stdio.h>
#include <string.h>

#include "scip/scip.h"
#include "scip/def.h"
#include "scip/scipshell.h"
#include "scip/scipdefplugins.h"
#include "scip/cons_countsols.h"
#include "scip/pub_nodepru.h"

#include "nodesel_oracle.h"
#include "nodesel_dagger.h"
#include "nodesel_policy.h"
#include "nodepru_oracle.h"
#include "nodepru_dagger.h"
#include "nodepru_policy.h"

//#ifdef SCIP6V
//#include "scip/scip_nodepru.h"
//SCIP_NODEPRU* SCIPgetNodepru(SCIP*);
//#endif

/* disable heuristics */ static
void disableHeurs(
   SCIP*                 scip,
   int                   freq 
   )
{
   int num_heur = 0;
   SCIP_HEUR** heurs = NULL;
   int i;

   num_heur = SCIPgetNHeurs( scip );
   heurs = SCIPgetHeurs( scip );
   for( i = 0; i < num_heur; i++ )
   {
      SCIPheurSetFreq( heurs[i], freq);
   }
}

/* disable separators */
static
void disableSepas(
   SCIP*                 scip,
   int                   freq 
   )
{
   int num_sepa = 0;
   SCIP_SEPA** sepas = NULL;
   int i;

   num_sepa = SCIPgetNSepas( scip );
   sepas = SCIPgetSepas( scip );
   for( i = 0; i < num_sepa; i++ )
   {
      SCIPsepaSetFreq( sepas[i], freq);
   }
}

static
SCIP_RETCODE fromCommandLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< input file name */
   const char*           solfname            /**< input file name */
   )
{
   SCIP_RETCODE retcode;
   const char* nodeselname;
   const char* nodepruname;
   SCIP_NODESEL* nodesel;
   SCIP_NODEPRU* nodepru;

   /********************
    * Problem Creation *
    ********************/

   /** @note The message handler should be only fed line by line such the message has the chance to add string in front
    *        of each message
    */
   SCIPinfoMessage(scip, NULL, "\n");
   SCIPinfoMessage(scip, NULL, "read problem <%s>\n", filename);
   SCIPinfoMessage(scip, NULL, "============\n");
   SCIPinfoMessage(scip, NULL, "\n");


   retcode = SCIPreadProb(scip, filename, NULL);

   switch( retcode )
   {
   case SCIP_NOFILE:
      SCIPinfoMessage(scip, NULL, "file <%s> not found\n", filename);
      return SCIP_OKAY;
   case SCIP_PLUGINNOTFOUND:
      SCIPinfoMessage(scip, NULL, "no reader for input file <%s> available\n", filename);
      return SCIP_OKAY;
   case SCIP_READERROR:
      SCIPinfoMessage(scip, NULL, "error reading file <%s>\n", filename);
      return SCIP_OKAY;
   default:
      SCIP_CALL( retcode );
   } /*lint !e788*/

   /*******************
    * Problem Solving *
    *******************/

   /* solve problem */
   SCIPinfoMessage(scip, NULL, "\nsolve problem\n");
   SCIPinfoMessage(scip, NULL, "=============\n\n");

   SCIP_CALL( SCIPsolve(scip) );

   if( solfname != NULL )
   {
      FILE* file = fopen(solfname, "w");
      SCIP_CALL( SCIPprintBestSol(scip, file, TRUE) );
      fclose(file);
   }

   /**************
    * Statistics *
    **************/

   SCIPinfoMessage(scip, NULL, "\nStatistics\n");
   SCIPinfoMessage(scip, NULL, "==========\n\n");

   SCIP_CALL( SCIPprintMyStatistics(scip, NULL) );

   /* node selector statistics */
   nodesel = SCIPgetNodesel(scip);
   nodeselname = SCIPnodeselGetName(nodesel);  
   if( strcmp(nodeselname, "policy") == 0 )
      SCIPnodeselpolicyPrintStatistics(scip, nodesel, NULL);
   else if( strcmp(nodeselname, "dagger") == 0 )
      SCIPnodeseldaggerPrintStatistics(scip, nodesel, NULL);

   /* node pruner statistics */
   nodepru = SCIPgetNodepru(scip);
   if( nodepru != NULL)
   {
      nodepruname = SCIPnodepruGetName(nodepru);  
      if( strcmp(nodepruname, "policy") == 0 )
         SCIPnodeprupolicyPrintStatistics(scip, nodepru, NULL);
      else if( strcmp(nodepruname, "dagger") == 0 )
         SCIPnodeprudaggerPrintStatistics(scip, nodepru, NULL);
   }

   return SCIP_OKAY;
}

/** evaluates command line parameters */
static
SCIP_RETCODE processShellArguments(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   argc,               /**< number of shell parameters */
   char**                argv,               /**< array with shell parameters */
   const char*           defaultsetname      /**< name of default settings file */
   )
{  /*lint --e{850}*/
   char* probname = NULL;
   char* settingsname = NULL;
   char* logname = NULL;
   char* vbcname = NULL;
   char* allsolsWritename = NULL;
   char* allsolsprefix = NULL;
   char* solfname = NULL;                    /**< input precomputed solution for training (oracle) */
   char* outputsolfname = NULL;              /**< output file to write the solution */
   char* nodeselname = NULL;
   char* nodeseltrj = NULL;
   char* nodepruname = NULL;
   char* nodeprutrj = NULL;
   char* nodeprupol= NULL;
   char* nodeselpol= NULL;
   char* probfeatsfname = NULL;
   char* snorm = NULL;
   char* pnorm = NULL;
   SCIP_Bool solrequired = FALSE;
   SCIP_Bool useSols = FALSE;
   SCIP_Bool quiet;
   int freq = 1;                             /**< frequency of heuristics and separators */ 
   SCIP_Longint nodelimit = -1;              /**< maximum number of nodes to process */
   SCIP_Real timelimit = -1;                 /**< maximum number of nodes to process */
   SCIP_Real margin=0;
   SCIP_Real beta=0;
   SCIP_Real ogapThreshold=0;
   SCIP_Real pru_noise=0;
   SCIP_Real sel_noise=0;
   float pscale = 1, sscale = 1;
   SCIP_Bool paramerror;
   int i;

   /********************
    * Parse parameters *
    ********************/

   quiet = FALSE;
   paramerror = FALSE;
   for( i = 1; i < argc; ++i )
   {
      if( strcmp(argv[i], "-l") == 0 )
      {
         i++;
         if( i < argc )
            logname = argv[i];
         else
         {
            printf("missing log filename after parameter '-l'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--allsols_prefix") == 0 )
      {
         i++;
         if( i < argc )
            allsolsprefix = argv[i];
         else
         {
            printf("missing allsolsprefix filename after parameter '--allsols_prefix'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--allsols_write") == 0 )
      {
         i++;
         if( i < argc )
            allsolsWritename = argv[i];
         else
         {
            printf("missing allsolsWrite filename after parameter '--allsols_write'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--useSols") == 0 ){
        useSols = 1;
      }
      else if( strcmp(argv[i], "--pnorm") == 0 )
      {
         i++;
         if( i < argc )
           pnorm = argv[i];
         else
         {
            printf("missing search network normalization file'--pnorm'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--snorm") == 0 )
      {
         i++;
         if( i < argc )
           snorm = argv[i];
         else
         {
            printf("missing search network normalization file'--snorm'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--vbc") == 0 )
      {
         i++;
         if( i < argc )
            vbcname = argv[i];
         else
         {
            printf("missing vbc filename after parameter '--vbc'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-q") == 0 ){
         quiet = TRUE;
      }
      else if( strcmp(argv[i], "-r") == 0 )
      {
         i++;
         if( i < argc )
            freq = atoi(argv[i]);
         else
         {
            printf("missing frequency after parameter '-r'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-n") == 0 )
      {
         i++;
         if( i < argc )
            nodelimit = atoll(argv[i]);
         else
         {
            printf("missing node limit after parameter '-n'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-m") == 0 )
      {
         i++;
         if( i < argc )
            margin = atof(argv[i]);
         else
         {
            printf("Margin for optimality check of continuous variables '-m'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-b") == 0 )
      {
         i++;
         if( i < argc )
            beta = atof(argv[i]);
         else
         {
            printf("Beta value '-b'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-t") == 0 )
      {
         i++;
         if( i < argc )
            timelimit = atof(argv[i]);
         else
         {
            printf("missing time limit after parameter '-t'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--pscale") == 0 )
      {
         i++;
         if( i < argc )
            pscale = atof(argv[i]);
         else
         {
            printf("Scale the weights of the pruner data points'--pscale'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--sscale") == 0 )
      {
         i++;
         if( i < argc )
            sscale = atof(argv[i]);
         else
         {
            printf("Scale the weight of the search network data points'--sscale'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-j") == 0 )
      {
         i++;
         if( i < argc )
            pru_noise = atof(argv[i]);
         else
         {
            printf("Noise to be added to the score of each node '-j'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-k") == 0 )
      {
         i++;
         if( i < argc )
            sel_noise = atof(argv[i]);
         else
         {
            printf("Noise to be added to the score of each node '-k'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-g") == 0 )
      {
         i++;
         if( i < argc )
            ogapThreshold = atof(argv[i]);
         else
         {
            printf("optimality gap treshold '-g'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-s") == 0 )
      {
         i++;
         if( i < argc )
            settingsname = argv[i];
         else
         {
            printf("missing settings filename after parameter '-s'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-f") == 0 )
      {
         i++;
         if( i < argc )
            probname = argv[i];
         else
         {
            printf("missing problem filename after parameter '-f'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "-o") == 0 )
      {
         i++;
         if( i < argc )
            solfname = argv[i];
         else
         {
            printf("missing optimal solution filename after parameter '-o'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--sol") == 0 )
      {
         i++;
         if( i < argc )
            outputsolfname = argv[i];
         else
         {
            printf("missing output solution filename after parameter '--sol'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--nodepru") == 0 )
      {
         i++;
         if( i < argc )
         {
            nodepruname = argv[i];

            if( strcmp(nodepruname, "oracle") == 0 || strcmp(nodepruname, "dagger") == 0 )
               solrequired = TRUE;

            if( strcmp(nodepruname, "policy") == 0 || strcmp(nodepruname, "dagger") == 0 )
            {
               i++;
               if( i < argc && argv[i][0] != '-' )
                  nodeprupol = argv[i];
               else
               {
                  printf("missing policy of node pruner '%s'\n", nodepruname);
                  paramerror = TRUE;
               }
            }
         }
         else
         {
            printf("missing node pruner name after parameter '--nodepru'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--nodeprutrj") == 0 )
      {
         i++;
         if( i < argc )
            nodeprutrj = argv[i];
         else
         {
            printf("missing node pruning trajectory filename after parameter '--nodeprutrj'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--nodesel") == 0 )
      {
         i++;
         if( i < argc )
         {
            nodeselname = argv[i];

            if( strcmp(nodeselname, "oracle") == 0 || strcmp(nodeselname, "dagger") == 0 )
               solrequired = TRUE;

            if( strcmp(nodeselname, "policy") == 0 || strcmp(nodeselname, "dagger") == 0 )
            {
               i++;
               if( i < argc && argv[i][0] != '-' )
                  nodeselpol = argv[i];
               else
               {
                  printf("missing policy of node selector '%s'\n", nodeselname);
                  paramerror = TRUE;
               }
            }
         }
         else
         {
            printf("missing node selector name after parameter '--nodesel'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--nodeseltrj") == 0 )
      {
         i++;
         if( i < argc )
            nodeseltrj = argv[i];
         else
         {
            printf("missing node selection trajectory filename after parameter '--nodeseltrj'\n");
            paramerror = TRUE;
         }
      }
      else if( strcmp(argv[i], "--probfeats") == 0 )
      {
         i++;
         if( i < argc )
            probfeatsfname = argv[i];
         else
         {
            printf("missing all node selection trajectory filename after parameter '--probfeats'\n");
            paramerror = TRUE;
         }
      }
      else
      {
         printf("invalid parameter <%s>\n", argv[i]);
         paramerror = TRUE;
      }
   }

   if( solrequired && solfname == NULL )
   {
      printf("missing optimal solution file\n");
      paramerror = TRUE;
   }

   if( !paramerror )
   {
      /***********************************
       * create log file message handler *
       ***********************************/

      if( quiet )
      {
         SCIPsetMessagehdlrQuiet(scip, quiet);
      }

      if( freq < 1 )
      {
         /* use most infeasible branching */
         SCIP_BRANCHRULE* branch_rule = NULL;
         branch_rule = SCIPfindBranchrule( scip, "mostinf" );
         SCIP_CALL( SCIPsetBranchrulePriority( scip, branch_rule, 9999999 ) );
         printf("Using most infeasible branching\n");

         disableHeurs(scip, freq);
         disableSepas(scip, freq);
         if( freq == 0 )
            printf("Using heuristics and separators only at root\n");
         else if( freq == -1 )
            printf("Disabled heuristics and separators\n");
      }

      if( timelimit > -1 )
      {
         SCIP_CALL( SCIPsetRealParam(scip, "limits/time", timelimit) );
         printf("Maximal time to run: %.2f\n", timelimit);
      }

      if( nodelimit > -1 )
      {
         SCIP_CALL( SCIPsetLongintParam(scip, "limits/totalnodes", nodelimit) );
         printf("Maximum number of total nodes to explore: %"SCIP_LONGINT_FORMAT"\n", nodelimit);
      }

      if( logname != NULL )
      {
         SCIPsetMessagehdlrLogfile(scip, logname);
      }

      if( vbcname != NULL )
      {
         SCIP_CALL( SCIPsetStringParam(scip, "vbc/filename", vbcname) );
      }

      if (allsolsWritename != NULL)
      {
         SCIP_CALL( SCIPsetStringParam(scip, "allsols/write", allsolsWritename) );
      }

      if (allsolsprefix != NULL)
      {
         SCIP_CALL( SCIPsetStringParam(scip, "allsols/prefix", allsolsprefix) );
      }

      if( nodepruname != NULL )
      {
         printf("include nodepru %s\n", nodepruname);
         if( strcmp(nodepruname, "oracle") == 0 )
         {
            SCIP_CALL( SCIPincludeNodepruOracle(scip) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/oracle/solfname", solfname) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/oracle/scale", pscale) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/oracle/margin", margin) );
            if( nodeprutrj != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/oracle/trjfname", nodeprutrj) );
            if( probfeatsfname != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/oracle/probfeatsfname", probfeatsfname) );
         }
         else if( strcmp(nodepruname, "dagger") == 0 )
         {
            SCIP_CALL( SCIPincludeNodepruDagger(scip) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/dagger/solfname", solfname) );
            SCIP_CALL( SCIPsetBoolParam(scip, "nodepruning/dagger/useSols", useSols) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/dagger/polfname", nodeprupol) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/dagger/ogapThreshold", ogapThreshold) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/dagger/score_noise", pru_noise) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/dagger/scale", pscale) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/dagger/margin", margin) );
            if (pnorm != NULL)
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/dagger/normfname", pnorm) );
            if( nodeprutrj != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/dagger/trjfname", nodeprutrj) );
            if( probfeatsfname != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/dagger/probfeatsfname", probfeatsfname) );
         }
         else if( strcmp(nodepruname, "policy") == 0 )
         {
            SCIP_CALL( SCIPincludeNodepruPolicy(scip) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/policy/polfname", nodeprupol) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodepruning/policy/score_noise", pru_noise) );
            if (pnorm != NULL)
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/policy/normfname", pnorm) );
            if( probfeatsfname != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodepruning/policy/probfeatsfname", probfeatsfname) );
         }
         else
         {
            printf("WARNING: unknown node pruner %s. ignored.\n", nodepruname);
         }
      }

      if( nodeselname != NULL )
      {
         SCIP_Bool ignored = FALSE;
         printf("include nodesel %s\n", nodeselname);
         if( strcmp(nodeselname, "oracle") == 0 )
         {
            SCIP_CALL( SCIPincludeNodeselOracle(scip) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/oracle/solfname", solfname) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/oracle/scale", sscale) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/oracle/margin", margin) );
            if( nodeseltrj != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/oracle/trjfname", nodeseltrj) );
            if( probfeatsfname != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/oracle/probfeatsfname", probfeatsfname) );
         }
         else if( strcmp(nodeselname, "dagger") == 0 )
         {
            SCIP_CALL( SCIPincludeNodeselDagger(scip) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/dagger/solfname", solfname) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/dagger/polfname", nodeselpol) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/dagger/ogapThreshold", ogapThreshold) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/dagger/score_noise", sel_noise) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/dagger/scale", sscale) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/dagger/margin", margin) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/dagger/beta", beta) );
            if (snorm != NULL)
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/dagger/normfname", snorm) );
            if( nodeseltrj != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/dagger/trjfname", nodeseltrj) );
            if( probfeatsfname != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/dagger/probfeatsfname", probfeatsfname) );
         }
         else if( strcmp(nodeselname, "policy") == 0 )
         {
            SCIP_CALL( SCIPincludeNodeselPolicy(scip) );
            SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/policy/polfname", nodeselpol) );
            SCIP_CALL( SCIPsetRealParam(scip, "nodeselection/policy/score_noise", sel_noise) );
            if (snorm != NULL)
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/policy/normfname", snorm) );
            if( probfeatsfname != NULL )
               SCIP_CALL( SCIPsetStringParam(scip, "nodeselection/policy/probfeatsfname", probfeatsfname) );
         }
         else
         {
            printf("WARNING: unknown node selector %s. ignored.\n", nodeselname);
            ignored = TRUE;
         }
         if( !ignored )
         {
            /* use specified node selector */
            SCIP_NODESEL* nodesel = SCIPfindNodesel( scip, nodeselname );
            SCIP_CALL( SCIPsetNodeselStdPriority( scip, nodesel, 9999999 ) );
         }
      }

      /***********************************
       * Version and library information *
       ***********************************/

      /*
      SCIPprintVersion(scip, NULL);
      SCIPinfoMessage(scip, NULL, "\n");

      SCIPprintExternalCodes(scip, NULL);
      SCIPinfoMessage(scip, NULL, "\n");
      */

      /*****************
       * Load settings *
       *****************/

      if( settingsname != NULL )
      {
         SCIP_CALL( SCIPreadParams(scip, settingsname) );
      }
      else if( defaultsetname != NULL )
      {
         SCIP_CALL( SCIPreadParams(scip, defaultsetname) );
      }

      /**************
       * Start SCIP *
       **************/

      if( probname != NULL )
      {
         SCIP_CALL( fromCommandLine(scip, probname, outputsolfname) );
      }
      else
      {
         return SCIP_OKAY;
      }

   }
   else{
      printf("\nsyntax: %s [-l <logfile>] [-q] [-s <settings>] [-f <problem>] [-o <solution>]\n"
         "  -l <logfile>  : copy output into log file\n"
         "  -q            : suppress screen messages\n"
         "  -s <settings> : load parameter settings (.set) file\n"
         "  -f <problem>  : load and solve problem file\n"
         "  -o <solution> : load optimal solution file\n",
         argv[0]);
   }

   return SCIP_OKAY;
}

/** creates a SCIP instance with default plugins, evaluates command line parameters, runs SCIP appropriately,
 *  and frees the SCIP instance
 */
static
SCIP_RETCODE runShell(
   int                        argc,               /**< number of shell parameters */
   char**                     argv,               /**< array with shell parameters */
   const char*                defaultsetname      /**< name of default settings  */
   )
{
   SCIP* scip = NULL;

   /*********
    * Setup *
    *********/

   /* initialize SCIP */
   SCIP_CALL( SCIPcreate(&scip) );
   
   /* include default SCIP plugins */
   SCIP_CALL( SCIPincludeDefaultPlugins(scip) );

   /**********************************
    * Process command line arguments *
    **********************************/
   SCIP_CALL( processShellArguments(scip, argc, argv, defaultsetname) );

   /********************
    * Deinitialization *
    ********************/

   SCIP_CALL( SCIPfree(&scip) );

   BMScheckEmptyMemory();

   return SCIP_OKAY;
}

int
main(
   int                        argc,
   char**                     argv
   )
{
   SCIP_RETCODE retcode;

   retcode = runShell(argc, argv, NULL);
   if( retcode != SCIP_OKAY )
   {
      SCIPprintError(retcode);
      return -1;
   }

   return 0;
}

