#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <onika/cpp_utils.h>

#include <vector>

namespace onika
{
  using namespace scg;

  namespace tutorial
  {

    static const std::vector<double> myvec = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };

    class DefaultValueFromCTorArgs : public OperatorNode
    {      
      // ========= I/O slots =======================
      // this demonstrates how initialization value can be replaced by a tuple of arguments as long as they are compatible with one of the slot type's constructor
      ADD_SLOT( std::vector<double> , input1, INPUT, std::make_tuple(size_t(10),5.0) );                   // initialization from size and initial value
      ADD_SLOT( std::vector<double> , input2, INPUT, std::make_tuple(myvec.begin()+1,myvec.begin()+6) );  // initialization from iterator pair
      ADD_SLOT( std::vector<double> , input3, INPUT, myvec ); // copy constructor, argument can be paced directly as default value, no need to place it in a tuple

    public:
      // Operator execution
      inline void execute () override final
      {      
        lout<<"input1 : length = " << input1->size()<<", content =";
        for(const auto& x:*input1) { lout << " " << x; }
        lout << std::endl;

        lout<<"input2 : length = " << input2->size()<<", content =";
        for(const auto& x:*input2) { lout << " " << x; }
        lout << std::endl;

        lout<<"input3 : length = " << input3->size()<<", content =";
        for(const auto& x:*input3) { lout << " " << x; }
        lout << std::endl;
      }

    };

    // === register factories ===  
    ONIKA_AUTORUN_INIT(default_value_from_constructor_args)
    {  
      OperatorNodeFactory::instance()->register_factory( "default_slot_value_from_ctor_args" , make_simple_operator< DefaultValueFromCTorArgs > );
    }

  }
}


