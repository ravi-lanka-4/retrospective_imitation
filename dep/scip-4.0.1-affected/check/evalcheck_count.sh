#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2017 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

export LANG=C

AWKARGS=""
FILES=""
for i in $@
do
    if test ! -e $i
    then
        AWKARGS="$AWKARGS $i"
    else
        FILES="$FILES $i"
    fi
done

for i in $FILES
do
    NAME=`basename $i .out`
    DIR=`dirname $i`
    OUTFILE=$DIR/$NAME.out
    RESFILE=$DIR/$NAME.res
    TEXFILE=$DIR/$NAME.tex
    PAVFILE=$DIR/$NAME.pav

    TSTNAME=`echo $NAME | sed 's/checkcount.\([a-zA-Z0-9_]*\).*/\1/g'`

    if test -f testset/$TSTNAME.test
    then
        TESTFILE=testset/$TSTNAME.test
    else
        TESTFILE=""
    fi

    # look for solufiles under the name of the test, the name of the test with everything after the first "_" stripped, and all
    SOLUFILE=""
    for f in $TSTNAME ${TSTNAME%%_*} all
    do
        if test -f testset/${f}.solu
        then
            SOLUFILE=testset/${f}.solu
            break
        fi
    done

    awk -f check_count.awk $AWKARGS $TESTFILE $SOLUFILE $OUTFILE | tee $RESFILE
done
