from types import SimpleNamespace

class ParamManager(object):
    def _init(self, **kw):
        return SimpleNamespace(**kw)

    def _post(self, params: SimpleNamespace):
        return vars(params)

    def merge(self, *dicts: dict) -> dict:
        merged_dict = {}
        for dic in dicts:
            merged_dict.update(dic)
        # print("merged_dict: ", merged_dict)
        return merged_dict
    
    def identity_kw(self, **kw):
        # for additional control
        p = self._init(**kw)
        return self._post(p)

    def set_demo(self):
        p = self._init()
        p.task_name = "demo"
        return self._post(p)

    def set_default(self):
        # set defalt with <store true>
        p = self._init()  
        p.zero_mean = None
        return self._post(p) 

    def set_toy1_subband(self):
        p = self._init()
        p.task_name = "toy1_subband"
        p.subband = None
        return self._post(p) 
    

if __name__ == "__main__":
    pm = ParamManager()
    params = pm.merge(
        pm.set_default,
        pm.set_demo(),
        pm.identity_kw(
            task_name = "demo2",
        ),
    )
    print(params)