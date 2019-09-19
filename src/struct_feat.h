/**@file   struct_feat.h
 * @brief  data structures for node features 
 * @author He He 
 *
 *  This file defines the interface for node feature implemented in C.
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_FEAT_H__
#define __SCIP_STRUCT_FEAT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "scip/def.h"

/** Features for node selector and pruner
 * Feature values are normalized accordingly.
 * The normalizer are all from the transformed problem,
 * since that is what actually used in the B&B tree.
 */
struct SCIP_Feat
{
   SCIP_Real*     vals;
   SCIP_Real      rootlpobj;
   SCIP_Real      sumobjcoeff;         /**< sum of coefficients of the objective */
   int            nconstrs;            /**< number of constraints of the problem */
   int            maxdepth;            /**< maximum depth of the B&B tree; use SCIP_Real since it's used as the divider */
   int            depth;
   SCIP_BOUNDTYPE boundtype;
   int            size;
   int            number;              /**< Node number used to mark feasible nodes */
   int            nonprobSize;         /**< Base features size */

   /* Normalization related data */
   int            norm_len;
   SCIP_Real*     norm_min;
   SCIP_Real*     norm_max;
};

#ifdef __cplusplus
}
#endif

#endif
