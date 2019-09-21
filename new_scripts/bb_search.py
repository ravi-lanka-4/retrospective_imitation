from __future__ import print_function
import glob, os
from datetime import datetime
import luigi, argparse
import shutil

suffix = '.lp'

probArg    = lambda probF : '-f %s '%probF
solArg     = lambda solF  : '-o %s '%solF
newSolArg  = lambda solF  : '--sol %s'%solF
sTrjArg    = lambda trjF  : '--nodeseltrj %s '%trjF
pScaleArg  = lambda pscale: '--pscale %s '%pscale
sScaleArg  = lambda sscale: '--pscale %s '%sscale

# Mode of SCIP DAgger
pOracleArg  = lambda : '--nodepru oracle '
sOracleArg  = lambda : '--nodesel oracle '
sDAggerArg  = lambda policyF: '--nodesel dagger %s '%policyF
sPolicyArg  = lambda policyF: '--nodesel policy %s '%policyF

# Misc arguments
# b - probability of using supervision on scaling up
# n - number of nodes to limit the search
miscArg  = lambda : '-b 0.3 '
normFArg = lambda normF: '--snorm %s '%normF
nodeCountArg = lambda count: '-n %s '%count

class Prepare(luigi.Task):
  # Must pass parameters
  workF    = luigi.Parameter()
  scratchF = luigi.Parameter(default='/tmp/')
  index    = luigi.IntParameter(default=0)

  def run(self):
    '''
    Prepares the folders for the run
    '''
    # Create the temporary directory
    workF = os.path.abspath(self.workF)
    filehandler = FileHandler(workF, scratch=self.scratchF)
    dataF   = filehandler.getDataF()
    solF    = filehandler.getSolF()
    logF    = filehandler.getLogF()
    trjF    = filehandler.getTrjF()
    policyF = filehandler.getPolicyF()

    # Create the data folder in the temporary path and softlink
    trainFexists = False
    for cF in glob.glob('%s/*'%workF):
      cc = os.path.basename(cF)
      trainFexists = trainFexists or ('train' in cc)
      #os.system('mkdir -p %s'%(dataF + '/%s'%cc))
      #os.system('mkdir -p %s'%(solF + '/%s'%cc))
      os.system('mkdir -p %s'%(logF + '/%s'%cc))
      os.system('mkdir -p %s'%(trjF + '/%s'%cc))

    # Create policy 
    os.system('mkdir -p %s'%(os.path.dirname(policyF)))

    # Copy solutions to temporary folder
    if not self.index: 
      # Link working problem files
      copyDirectory(workF, dataF)

      # Link the sol files
      solF = filehandler.getSolF(tmp=False)
      newSolF = filehandler.getSolF()
      copyDirectory(solF, newSolF)

    else:
      # unlabeled file handler
      ufh = FileHandler(workF + '/unlabeled%d/'%(self.index-1), scratch=self.scratchF)
      tfh = FileHandler(workF + '/train/', scratch=self.scratchF)

      # Copy sols
      sol_unlabeled = ufh.getSolF()
      sol_train = tfh.getSolF()
      os.system('cp -R %s %s'%(sol_unlabeled, sol_train))

      # Copy data foles
      dataF_unlabeled = ufh.getDataF()
      dataF_train = tfh.getDataF()
      os.system('cp -R %s %s'%(dataF_unlabeled, dataF_train))

    if not trainFexists:
      cc = 'train'
      os.system('mkdir -p %s'%(dataF + '/%s'%cc))
      os.system('mkdir -p %s'%(solF + '/%s'%cc))
      os.system('mkdir -p %s'%(logF + '/%s'%cc))
      os.system('mkdir -p %s'%(trjF + '/%s'%cc))

    return

  def complete(self):
    workF = os.path.abspath(self.workF)
    if not self.index:
      filehandler = FileHandler(workF + '/train/', scratch=self.scratchF)
    else:
      filehandler = FileHandler(workF + '/unlabeled%d/'%(self.index), scratch=self.scratchF)

    solF = filehandler.getSolF() 
    if not os.path.isdir(solF):
      return False
    else:    
      return (len(os.listdir(solF)) != 0)

class SCIPInferenceInstance(luigi.Task):
  task_namespace = 'SCIPInferenceInstance'

  # Must pass parameters
  probF = luigi.Parameter()
  index = luigi.IntParameter(default=0)
  nnodes = luigi.IntParameter(default=500)
  accumulateSols = luigi.IntParameter(default=0)

  def run(self):

    # Current class of label
    cc = os.path.basename(os.path.dirname(self.probF))
    filehandler = FileHandler(self.probF, '/tmp/')
    if not self.accumulateSols:
      newSolF = filehandler.getNewSolF(tmp=False)
    else:
      newSolF = filehandler.getNewSolF(index=self.index, tmp=False)

    logF    = filehandler.getLogF(tmp=False)
    trjF    = filehandler.getTrjF(tmp=False)

    # Get the previous iteration policy for running in DAgger mode
    policyF = filehandler.getPolicyF(self.index, tmp=False)
    normF   = filehandler.getNormF(self.index, tmp=False)
    param   = sPolicyArg(policyF)                    # Policy with file name
    param  += probArg(self.probF)                    # Input file
    param  += normFArg(normF)                        # Normalization file
    param  += nodeCountArg(self.nnodes)           # Misc arguments
    param  += newSolArg(newSolF)                     # New sol file

    # Run policy mode
    os.system('../bin/scipdagger -s scip.set ' + param + ' > %s'%logF)
    return

  def complete(self):
    cc = os.path.basename(os.path.dirname(self.probF))
    filehandler = FileHandler(self.probF, '/tmp/')
    if not self.accumulateSols:
      newSolF = filehandler.getNewSolF(tmp=False)
    else:
      newSolF = filehandler.getNewSolF(index=self.index, tmp=False)
    return os.path.isfile(newSolF)

class SCIPDAggerInstance(luigi.Task):
  task_namespace = 'SCIPDAggerInstance'
  pruneScale = luigi.IntParameter(default=2)
  searchScale = luigi.IntParameter(default=2)

  # Must pass parameters
  probF     = luigi.Parameter()
  scratchF  = luigi.Parameter()
  index     = luigi.IntParameter(default=0)
  nnodes    = luigi.IntParameter(default=500)

  def run(self):
    # Current class of label
    filehandler = FileHandler(self.probF, self.scratchF)
    solF   = filehandler.getSolF(tmp=False)
    logF   = filehandler.getLogF(tmp=False)
    trjF   = filehandler.getTrjF(tmp=False)

    # Compile the arguments
    if self.index < 0:
      param = sOracleArg()
    else:
      # Get the previous iteration policy for running in DAgger mode
      policyF = filehandler.getPolicyF(self.index, tmp=False)
      param = sDAggerArg(policyF)
      param += miscArg()
      param += nodeCountArg(self.nnodes)

      # Normalization file
      normF = filehandler.getNormF(self.index, tmp=False)
      param += normFArg(normF)

    # Use pruning policy only for the first iteration
    if self.index < 0:
      param += pOracleArg()

    param += pScaleArg(self.pruneScale)
    param += sScaleArg(self.searchScale)
    param += probArg(self.probF)
    param += solArg(solF)
    param += sTrjArg(trjF)

    # Run policy mode
    os.system('../bin/scipdagger -s scip.set ' + param + ' > %s'%logF)
    return

  def complete(self):
    filehandler = FileHandler(self.probF, self.scratchF)
    trjF = filehandler.getTrjF(tmp=False)
    return (os.path.isfile(trjF + '.1') and os.path.isfile(trjF + '.2')) 

def SCIPCompileTraj(workF, index, clear=True):
  # Compile the trajectories together
  cF1 = os.path.join(workF, 'current%d.1'%index)
  cF2 = os.path.join(workF, 'current%d.2'%index)
  wF  = os.path.join(workF, 'current%d.weight'%index)
  os.system('touch %s'%(cF1))
  os.system('touch %s'%(cF2))
  os.system('touch %s'%(wF))

  completedFlag = {}
  for ctrj in glob.glob(workF + '/*trj*'):
    cF = os.path.basename(ctrj.split('.')[0])
    if cF in completedFlag:
      continue
    else:
      cc = ctrj.strip('.1').strip('.2').strip('.weight')
      os.system('cat %s >> %s'%(cc + '.1', cF1))
      os.system('cat %s >> %s'%(cc + '.2', cF2))
      os.system('cat %s >> %s'%(cc + '.weight', wF))
      completedFlag[cF] = True

  if clear:
    for ctrj in glob.glob(workF + '/*trj*'):
      os.system('rm %s'%ctrj)

  return

class SCIPDAgger(luigi.Task):
  task_namespace = 'SCIPDAgger'

  # Must pass parameters
  probF    = luigi.Parameter()
  scratchF = luigi.Parameter('/tmp/')
  nnodes   = luigi.IntParameter(default=500)

  # Index determines the mode
  # 0 - Oracle mode - supervised
  # >= 1 - Policy mode (DAgger)
  index = luigi.IntParameter(default=0)

  def output(self):
    cprobF = os.path.abspath(self.probF)
    filehandler = FileHandler(cprobF, self.scratchF)
    policyF = filehandler.getPolicyF(self.index)
    normF   = filehandler.getNormF(self.index)
    return [luigi.LocalTarget(policyF), luigi.LocalTarget(normF)]

  def run(self):
    # Compile all trajectories together
    cprobF = os.path.abspath(self.probF)
    filehandler = FileHandler(cprobF, self.scratchF)
    trjF = filehandler.getTrjF()
    policyF, normF = self.output()
    policyF, normF = policyF.path, normF.path

    # Compile the logs 
    validF = os.path.join(trjF, 'valid')
    SCIPCompileTraj(workF=validF, index=self.index)
    trainF = os.path.join(trjF, 'train')
    SCIPCompileTraj(workF=trainF, index=self.index)
    #testF  = os.path.join(trjF, 'test')
    #SCIPCompileTraj(workF=testF, index=self.index)

    # Normalize the input data 
    tc = lambda x: "%s"%x + "/current%d"%(self.index)
    nc = lambda x: "%s"%x + "/current%d.norm"%(self.index)
    wc = lambda x: "%s"%x + "/current%d.weight"%(self.index)
    os.system('python ../pyscripts/normdata.py --i %s --o %s --n %s'%(tc(trainF), nc(trainF), normF))
    os.system('python ../pyscripts/normdata.py --i %s --o %s --p %s'%(tc(validF), nc(validF), normF))
    #os.system('python ../pyscripts/normdata.py --i %s --o %s --p %s'%(tc(testF), nc(testF), normF))

    # Train the policies
    os.system('python ../pyscripts/searchnet.py --v %s --t %s --m %s --w %s --x %s'\
                    %(nc(validF), nc(trainF), policyF, wc(trainF), wc(validF)))

    return

  def requires(self):
    cprobF = os.path.abspath(self.probF)
    fh  = FileHandler(cprobF, self.scratchF)
    cpf = fh.getDataF()
    for cc in ['/train/', '/valid/']:
      for cprob in glob.glob(cpf + '/%s/*%s'%(cc, suffix)):
        yield SCIPDAggerInstance(probF=cprob, index=self.index-1, scratchF=self.scratchF, nnodes=self.nnodes)

    return

  def complete(self):
    modelF, normF = self.output()
    return os.path.isfile(modelF.path) and os.path.isfile(normF.path)

class SCIPInference(luigi.Task):
  task_namespace = 'SCIPInference'
  scratchF = luigi.Parameter('/tmp/')

  # Must pass parameters
  probF = luigi.Parameter()
  index = luigi.IntParameter(default=0)
  accumulateSols = luigi.IntParameter(default=0)
  nnodes = luigi.IntParameter(default=500)

  def run(self):
    return

  def requires(self):
    cprobF = os.path.abspath(self.probF)
    fh  = FileHandler(cprobF, self.scratchF)
    cpf = fh.getDataF()
    for cprob in glob.glob(cpf + '/*%s'%(suffix)):
        yield SCIPInferenceInstance(probF=cprob, index=self.index, accumulateSols=self.accumulateSols, nnodes=self.nnodes)

    return 

  def complete(self):
    cprobF = os.path.abspath(self.probF)
    filehandler = FileHandler(cprobF, '/tmp/')
    if not self.accumulateSols:
      newSolF = filehandler.getNewSolF()
    else:
      newSolF = filehandler.getNewSolF(self.index)
    nsols = len(os.listdir(newSolF))
    return nsols != 0

####
# Helper functions -- begin
####

class FileHandler:
  '''
  File names handler
  '''
  def __init__(self, inpF, scratch='', suffix='lp'):
    if inpF[-1] == '/':
      self.inpF   = inpF + '/'
      self.suffix = suffix
      self.isfile = False
    elif ('.%s'%suffix in inpF):
      self.inpF = inpF
      _, self.suffix  = os.path.splitext(self.inpF)
      self.isfile = True
    else:
      # For now treat anything else as a folder
      self.inpF   = inpF + '/'
      self.suffix = suffix
      self.isfile = False
      #raise Exception('File or Folder does not exist')

    self.scratch = scratch
    if len(self.scratch):
        self.scratch = self.scratch + '/'

    self.suffix  = self.suffix.strip('.')
    return

  def getBaseF(self):
    return os.path.basename(self.inpF)

  def getDataF(self, tmp=True):
    if tmp:
      return self.scratch + self.inpF
    else:
      return self.inpF

  def getPolicyF(self, index=0, tmp=True):
    cF = self.inpF.replace('/%s/'%(self.suffix), '/policy/')
    if self.isfile:
      # Since the files have train/test/valid appended to their name
      cF = os.path.dirname(cF)

    if tmp:
      return self.scratch + os.path.dirname(cF) + '/searchPolicy.%d.h5'%(index)
    else:
      return os.path.dirname(cF) + '/searchPolicy.%d.h5'%(index)

  def getNormF(self, index, tmp=True):
    cF = self.inpF.replace('/%s/'%(self.suffix), '/policy/')
    if self.isfile:
      # Since the files have train/test/valid appended to their name
      cF = os.path.dirname(cF)

    if tmp:
      return self.scratch + os.path.dirname(cF) + '/searchPolicy.%d.norm'%(index)
    else:
      return os.path.dirname(cF) + '/searchPolicy.%d.norm'%(index)

  def getSolF(self, tmp=True):
    cF = self.inpF.replace('/%s/'%(self.suffix), '/sol/')
    if tmp:
      return self.scratch + cF.replace('.' + self.suffix, '.sol')
    else:
      return cF.replace('.' + self.suffix, '.sol')

  def getNewSolF(self, index=None, tmp=True):
    if index != None:
      cF = self.inpF.replace('/%s/'%(self.suffix), '/sol%d/'%index)
    else:
      cF = self.inpF.replace('/%s/'%(self.suffix), '/sol/')

    if tmp:
      out = self.scratch + cF.replace('.' + self.suffix, '.sol')
    else:
      out = cF.replace('.' + self.suffix, '.sol')

    # Check if the folder exists
    os.system('mkdir -p %s'%os.path.dirname(out))
    return out

  def getTrjF(self, tmp=True):
    cF = self.inpF.replace('/%s/'%(self.suffix), '/trj/')
    if tmp:
      return self.scratch + cF.replace('.' + self.suffix, '.trj')
    else:
      return cF.replace('.' + self.suffix, '.trj')

  def getLogF(self, tmp=True):
    cF = self.inpF.replace('/%s/'%(self.suffix), '/log/')
    if tmp:
      return self.scratch + cF.replace('.' + self.suffix, '.log')
    else:
      return cF.replace('.' + self.suffix, '.log')

 
def copyDirectory(src, dest):
  try:
    shutil.copytree(src, dest)
    # Directories are the same
  except shutil.Error as e:
    print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
  except OSError as e:
    print('Directory not copied. Error: %s' % e)
  
####
# Helper functions -- end
####

def preproc(inpF, policyF, scratchF, remove=False):
  # New policy
  fh = FileHandler(inpF, scratchF)
  newPolicy = fh.getPolicyF(0)
  newNorm   = fh.getNormF(0)

  # Policy from previous iteration
  fh = FileHandler(policyF, scratchF)
  oldPolicy = fh.getPolicyF(totalItr-1)
  oldNorm   = fh.getNormF(totalItr-1)

  # Copy policy and norm
  if not remove:
    os.system('mkdir -p %s'%os.path.dirname(newPolicy))
    os.system('cp %s %s'%(oldPolicy, newPolicy))
    os.system('mkdir -p %s'%os.path.dirname(newNorm))
    os.system('cp %s %s'%(oldNorm, newNorm))
  else:
    # To be removed so luigi does run the policy generation using oracle method
    os.system('rm %s'%(newPolicy))
    os.system('rm %s'%(newNorm))

  return

def cmdLineParser():
  '''
  Command Line Parser.
  '''
  parser = argparse.ArgumentParser(description='Retrospective Imitation')
  parser.add_argument('-i','--input', dest='inputF', type=str,
                      action='store', default='/Users/subrahma/proj/retro_imitation/scip-dagger/data/mvc_test_retro/mvc_test_a/',
                      help='Input file for processing')
  parser.add_argument('-p','--policy', dest='policyF', type=str,
                      action='store', default=None,
                      help='Policy Folder')
  parser.add_argument('-t','--scratch', dest='scratchF', type=str,
                      action='store', default='/tmp/',
                      help='Scratch Folder')
  parser.add_argument('-n','--num_nodes', dest='numNodes', type=int, default=100,
                      action='store', help='Number of nodes')
  parser.add_argument('-ti','--total_iterations', dest='totalItr', type=int, default=3,
                      action='store', help='Total iterations')
  parser.add_argument('-d','--dagger_iterations', dest='dagItr', type=int, default=2,
                      action='store', help='Total iterations')
  parser.add_argument('-s','--suffix', dest='suffix', type=str, default='.lp',
                      action='store', help='suffix')

  return parser.parse_args()

if __name__ == '__main__':
  args = cmdLineParser()
  inpF = args.inputF
  policyF  = args.policyF
  scratchF = args.scratchF
  numNodes = args.numNodes
  suffix   = args.suffix
  totalItr = args.totalItr
  dagItr   = args.dagItr

  inpF = inpF + '/%s/'%suffix.strip('.')

  # Prepare the input for each iteration
  luigi.build([Prepare(workF=inpF, scratchF=scratchF, index=0)], workers=1, local_scheduler=True)

  if policyF:
    if os.path.isfile(policyF):
        raise Exception('File does not exist')

    # Prepare the input for retrospective imitation
    policyF = policyF + '/%s/'%suffix.strip('.')
    preproc(inpF, policyF, scratchF)

    # Inference on unseen data
    inference_folder = '%s/unlabeled0/'%(inpF)
    valid_folder     = '%s/valid/'%inpF
    luigi.build([SCIPInference(probF=inference_folder, index=0, nnodes=numNodes)], workers=1, local_scheduler=True)
    luigi.build([SCIPInference(probF=valid_folder, index=0, nnodes=numNodes)], workers=1, local_scheduler=True)

    # Update unlabeled0 to train on for the policies
    luigi.build([Prepare(workF=inpF, scratchF=scratchF, index=1)], workers=1, local_scheduler=True)

    # Now remove the policies and proceed as usual
    preproc(inpF, policyF, scratchF, remove=True)

    # Start from second unlabeled folder
    unlabeledIdx = 2
  else:
    # Start from second unlabeled folder
    unlabeledIdx = 1

  ### 
  # 0 -> Oracle mode
  # 1, 2 -> DAgger 
  # End of iteration 2 -> Add unlabeled0 data and improve policies
  # 3, 4 -> DAgger on unlabeled0 + labeled
  # End of iteration 4 -> Add unlabeled1 data and improve policies
  ###

  for i in range(0, totalItr):
    # Run DAgger -- first iteration is oracle mode
    luigi.build([SCIPDAgger(probF=inpF, index=i, nnodes=numNodes)], workers=1, local_scheduler=True)

    if (i%dagItr == 0) and (i != 0):
      # Inference on unseen data
      inference_folder = '%s/unlabeled%d/'%(inpF, unlabeledIdx-1)
      luigi.build([SCIPInference(probF=inference_folder, index=i, nnodes=numNodes)], workers=1, local_scheduler=True)

      # Prepare the input for each iteration -- this version copies the unlabeled data with solutions
      luigi.build([Prepare(workF=inpF, scratchF=scratchF, index=unlabeledIdx)], workers=1, local_scheduler=True)

      # Increment the counter for retro unlabeled data
      unlabeledIdx = unlabeledIdx + 1


