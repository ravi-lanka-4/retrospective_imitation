from bb_search import *

nodesArg = lambda nnodes: '-n %d '%nnodes
timeArg = lambda ntimes: '-t %f '%ntimes
miscArg = lambda : '-s scip.set '

class LogHandler:
  def __init__(self, logF):
    assert os.path.isfile(logF), "Log file missing %s"%logF
    self.logF = logF
    return

  def proc(self, func):
    with open(self.logF, 'r') as F:
        data = F.readlines()

    for cline in data: 
      if func(cline):
        return cline

    return None

  def getNodes(self):
    cmatch = 'Solving Nodes'
    cstr = self.proc(lambda x: cmatch in x)
    if cstr:
      return float(cstr.strip(cmatch).strip('/n').strip(':'))
    else:
      return None

  def getTime(self):
    cmatch = 'Solving Time (sec)'
    cstr = self.proc(lambda x: cmatch in x)
    if cstr:
      return float(cstr.strip(cmatch).strip('/n').strip(':'))
    else:
      return None

class SCIPReference(luigi.Task):
  task_namespace = 'SCIPReference'
  scratchF = luigi.Parameter('/tmp/')

  # Must pass parameters
  probF = luigi.Parameter()
  testF = luigi.Parameter()
  index = luigi.IntParameter(default=0)
  accumulateSols = luigi.IntParameter(default=0)
  numNodes = luigi.IntParameter(default=None)
  runScip  = luigi.BoolParameter(default=True)
  
  def run(self):
    # Used for the file name
    cprobF = os.path.abspath(self.testF)
    fh  = FileHandler(cprobF, self.scratchF)
    cpf = fh.getDataF()

    for cprob in glob.glob(cpf + '/*%s'%(suffix)):
      # get log file
      cfl   = FileHandler(cprob, '')
      logF  = cfl.getLogF()
      baseF = cfl.getBaseF()

      # sol file name
      solfh = FileHandler(self.probF + '/' + baseF, self.scratchF)
      solF  = solfh.getNewSolF(index=self.index)

      #####
      # Prepare the parameters
      #####

      # Create log handler
      lh = LogHandler(logF)
      params = ""
      time = None
      nnodes = None
      if not self.numNodes:
        # time limit
        time    = lh.getTime()
        if not time:
          # Error with parsing
          continue
        params += timeArg(time)
      else:
        # node limit
        nnodes  = lh.getNodes()
        if not nnodes:
          # Error with parsing
          continue
        params += nodesArg(nnodes)

      if self.runScip:
          # SCIP reference
          # Get the previous iteration policy for running in DAgger mode
          params += miscArg()
          params += probArg(cprob)
          params += newSolArg(solF)

          os.system('../bin/scipdagger %s'%(params))
      else:
          # Gurobi reference
          if not nnodes:
            os.system('gurobi_cl ResultFile=%s TimeLimit=%f Threads=1 %s'%(solF, nnodes, cprob))
          else:
            os.system('gurobi_cl ResultFile=%s NodeLimit=%d Threads=1 %s'%(solF, nnodes, cprob))

    return 

  def complete(self):
    return False
    cprobF = os.path.abspath(self.probF)
    filehandler = FileHandler(cprobF, '/tmp/')
    newSolF = filehandler.getNewSolF(self.index)
    nsols = len(os.listdir(newSolF))
    return nsols != 0

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

  for i in range(0, totalItr):
    # First run on test data -- logs are updated in the log file
    test_folder = '%s/test/'%(inpF)
    luigi.build([SCIPInference(probF=test_folder, index=i, accumulateSols=1, nnodes=numNodes)], workers=1, local_scheduler=True)

    # SCIP performance with num of nodes or run time
    scip_folder = '%s/scip/'%(inpF)
    luigi.build([SCIPReference(probF=scip_folder, testF=test_folder, index=i, accumulateSols=1, numNodes=numNodes)], 
                 workers=1, local_scheduler=True)

    # Gurobi performance with nodes or run time
    gurobi_folder = '%s/gurobi/'%(inpF)
    luigi.build([SCIPReference(probF=gurobi_folder, testF=test_folder, index=i, accumulateSols=1, numNodes=numNodes, runScip=False)], 
                 workers=1, local_scheduler=True)



