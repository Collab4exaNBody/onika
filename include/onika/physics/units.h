/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <limits>
#include <cmath>
#include <string>

#include <onika/physics/constants.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>

#define EXANB_UNITS_V2 1
#define EXANB_UNITS_DEPRECATED [[deprecated]]

#ifdef EXANB_UNITS_DEPRECATED
#define EXANB_LEGACY_UNITS_DEPRECATED 1
#else
#define EXANB_UNITS_DEPRECATED /**/
#endif

#define ONIKA_SI_UNIT_SYSTEM meter,kilogram,second,coulomb,kelvin,mol,candela,radian

#ifndef ONIKA_INTERNAL_UNIT_SYSTEM
#define ONIKA_INTERNAL_UNIT_SYSTEM ONIKA_SI_UNIT_SYSTEM
#endif

// exaSPH
// meter,gram,second,elementary_charge,kelvin,particle,candela,radian

namespace onika
{

  namespace physics
  {

    /*
     * Defines different kinds of units.
     * ENERGY is a special kind, not stored in quantities, as it is converted to length^2 * mass / time^2
     */
    enum UnitClass
    {
      LENGTH,
      MASS,
      TIME,
      CHARGE,
      TEMPERATURE,
      AMOUNT,
      LUMINOSITY,
      ANGLE,
      ENERGY,
      NUMBER_OF_UNIT_CLASSES = ENERGY,
      OTHER = -1
    };
    
    /*
     * unit classes as strings
     */
    static inline constexpr const char* g_unit_class_str[NUMBER_OF_UNIT_CLASSES+1] = {
      "length",
      "mass",
      "time",
      "charge",
      "temperature",
      "amount",
      "luminosity",
      "angle",
      "energy"
    };
    
    // shortcuts to particular values
    static inline constexpr auto elementaryChargeCoulomb = elementaryCharge;
    static inline constexpr auto undefined_value = std::numeric_limits<double>::quiet_NaN();
    
    /*
     * A unit defined by its class (length, mass, etc.) ,
     * its conversion factor to its IS system counterpart
     * and its full and short name.
     */
    struct UnitDefinition
    {
      UnitClass m_class = OTHER;
      double m_to_si = 1.0;
      const char* m_short_name = "?";
      const char* m_name = "unknow";
      ONIKA_HOST_DEVICE_FUNC inline bool operator == (const UnitDefinition& other) const { return m_class==other.m_class && m_to_si==other.m_to_si; }
      inline const char* short_name() const { return m_short_name; }
      inline const char* name() const { return m_name; }
    };

    /*********************************************************************/
    /***************** All units available *******************************/
    /*********************************************************************/

    // length
    static inline constexpr UnitDefinition meter              = { LENGTH     , 1.0                     , "m"        , "meter" };
    static inline constexpr UnitDefinition millimeter         = { LENGTH     , 1.0e-3                  , "mm"       , "millimeter" };
    static inline constexpr UnitDefinition micron             = { LENGTH     , 1.0e-6                  , "um"       , "micron" };
    static inline constexpr UnitDefinition nanometer          = { LENGTH     , 1.0e-9                  , "nm"       , "nanometer" };
    static inline constexpr UnitDefinition angstrom           = { LENGTH     , 1.0e-10                 , "ang"      , "angstrom" };
    
    // mass
    static inline constexpr UnitDefinition kilogram           = { MASS       , 1.0                     , "kg"       , "kilogram" };
    static inline constexpr UnitDefinition gram               = { MASS       , 1.0e-3                  , "g"        , "gram" };
    static inline constexpr UnitDefinition atomic_mass_unit   = { MASS       , 1.660539040e-27         , "Da"       , "Dalton" };
    
    // time
    static inline constexpr UnitDefinition hour               = { TIME       , 3600                    , "h"        , "hour" };
    static inline constexpr UnitDefinition second             = { TIME       , 1.0                     , "s"        , "second" };
    static inline constexpr UnitDefinition microsecond        = { TIME       , 1.0e-6                  , "us"       , "microsecond" };
    static inline constexpr UnitDefinition nanosecond         = { TIME       , 1.0e-9                  , "ns"       , "nanosecond" };
    static inline constexpr UnitDefinition picosecond         = { TIME       , 1.0e-12                 , "ps"       , "picosecond" };
    static inline constexpr UnitDefinition fetosecond         = { TIME       , 1.0e-15                 , "fs"       , "femtosecond" };

    // Electric current
    static inline constexpr UnitDefinition coulomb            = { CHARGE     , 1.0                     , "C"        , "coulomb" };
    static inline constexpr UnitDefinition elementary_charge  = { CHARGE     , elementaryChargeCoulomb , "e-"       , "elementary_charge" };

    // temperature
    static inline constexpr UnitDefinition kelvin             = { TEMPERATURE, 1.0                     , "K"        , "kelvin" };
    static inline constexpr UnitDefinition celsius            = { TEMPERATURE, 1.0                     , "°"        , "celsius" }; // should never be used

    // amount of substance
    static inline constexpr UnitDefinition mol                = { AMOUNT     , 1.0                     , "mol"      , "mol" };
    static inline constexpr UnitDefinition particle           = { AMOUNT     , 1.0e-23 / 6.02214076    , "particle" , "particle" }; 

    // luminosity
    static inline constexpr UnitDefinition candela            = { LUMINOSITY , 1.0                     , "cd"       , "candela" };

    // angle
    static inline constexpr UnitDefinition radian             = { ANGLE      , 1.0                     , "rad"      , "radian" };
    static inline constexpr UnitDefinition degree             = { ANGLE      , M_PI/180.0              , "degree"   , "degree" };

    // energy
    static inline constexpr UnitDefinition joule              = { ENERGY     , 1.0                     , "J"        , "joule" };
    static inline constexpr UnitDefinition electron_volt      = { ENERGY     , elementaryChargeCoulomb , "eV"       , "electron_volt" };
    static inline constexpr UnitDefinition calorie            = { ENERGY     , 4.1868                  , "cal"      , "calorie" };
    static inline constexpr UnitDefinition kcalorie           = { ENERGY     , 4186.8                  , "kcal"     , "kcalorie" };
    
    // unit less
    static inline constexpr UnitDefinition no_unity           = { OTHER      , 1.0                     , ""         , "" };

    // unknown
    static inline constexpr UnitDefinition unknown            = { OTHER      , undefined_value         , ""         , "unknow" };

    /*********************************************************************/
    /*********************************************************************/

    /*
     * A UnitSystem is a set of units, one per type (class) of unit
     */
    struct UnitSystem
    {
      UnitDefinition m_units[NUMBER_OF_UNIT_CLASSES+1];      
      inline const UnitDefinition& length() const { return m_units[LENGTH]; }
      inline const UnitDefinition& time() const { return m_units[TIME]; }
      inline const UnitDefinition& charge() const { return m_units[CHARGE]; }
      inline const UnitDefinition& temperature() const { return m_units[TEMPERATURE]; }
      inline const UnitDefinition& amount() const { return m_units[AMOUNT]; }
      inline const UnitDefinition& luminosity() const { return m_units[LUMINOSITY]; }
      inline const UnitDefinition& angle() const { return m_units[ANGLE]; }
      inline const UnitDefinition& energy() const { return m_units[ENERGY]; }
    };

    /*
     * powers associated with units of each class 
     */
    struct UnitPowers
    {
      double m_powers[NUMBER_OF_UNIT_CLASSES+1] = { 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. };
    };
      
    // list of all available unit definitions
    static inline constexpr UnitDefinition all_units[] = {
      meter, millimeter, micron, nanometer, angstrom,
      kilogram, gram, atomic_mass_unit,
      hour, second, microsecond, nanosecond, picosecond, fetosecond,
      coulomb, elementary_charge,
      kelvin, celsius,
      mol, particle,
      candela,
      radian, degree,
      joule, electron_volt, calorie, kcalorie };
    
    // number of available unit definitions 
    static inline constexpr int number_of_units = sizeof(all_units) / sizeof(UnitDefinition);
    
    
    /*
     * return a unit definition given a unit's short name
     */
    UnitClass unit_class_from_name(const std::string& s);
    UnitDefinition unit_from_name(const std::string& s);
    UnitDefinition unit_from_symbol(const std::string& s);

    /*
     * International System units
     */
    static inline constexpr UnitSystem SI = { { ONIKA_SI_UNIT_SYSTEM } };
  
    /*
     * Internal unit system, as defined by application via ONIKA_INTERNAL_UNIT_SYSTEM predefined macro
     */
    void set_internal_unit_system( const UnitSystem & ius );
    const UnitSystem& internal_unit_system();

    /*
     * A Quantity is a value expressed with a certain set of units (at most one unit per unit class) and associated powers
     * Quantity has an implicit conversion operator to double, wich converts expressed value to application's internal unit system
     * Exemple :
     * using namespace onika::physics;
     * Quantity q = ONIKA_QUANTITY( 3.2 * (g^3) / s ); // ONIKA_QUANTITY macro allows to use short name unit definitons in online expressions
     * Quantity r = 3.2 * (gram^3) / second; // same as above
     * double a = q;
     * double b = q.convert(); // same as above
     * double c = q.convert( internal_unit_system ); // same as above
     * double x = q.convert( SI ); // converts to Internal System units based value
     *
     * Note :
     * to enable constexpr quantity without using external libs (like GCEM)
     * or futuristic C++26 standard which enables constexpr math functions,
     * we make the strong assumtion that unit powers cannot be fractional, even though they're stored as double
     * thus, we only need constexpr a pow that computes integral powers, which is implemented in onika::cuda::pow_ce_dint
     */
    struct Quantity
    {
      double m_value = 0.0;
      UnitSystem m_system = SI;
      UnitPowers m_unit_powers = { { 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. } };

      ONIKA_HOST_DEVICE_FUNC inline constexpr double convert( const UnitSystem& other ) const
      {
        using onika::cuda::pow_ce_dint;
        double value = m_value;
        for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
        {
          if( m_system.m_units[i].m_class != i ) // invalid units or internal error
          {
            return undefined_value;
          } 
          if( m_system.m_units[i].m_to_si != other.m_units[i].m_to_si )
          {
            value *= pow_ce_dint( m_system.m_units[i].m_to_si / other.m_units[i].m_to_si , m_unit_powers.m_powers[i] );
          }
        }
        return value;      
      }
      ONIKA_HOST_DEVICE_FUNC inline double convert() const
      {
        return convert( internal_unit_system() );
      }
      
      // ONIKA_HOST_DEVICE_FUNC inline operator double() const { return convert(); }
    };

    /*
     * Multiplies 2 quantities.
     * Converts qrhs units to the units used in qlhs if needed.
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator * ( const Quantity& qlhs , const Quantity& qrhs )
    {
      using onika::cuda::pow_ce_dint;
      Quantity q = qlhs;
      q.m_value *= qrhs.m_value;
      for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
      {
        if( ( q.m_system.m_units[i].m_class == i ) && ( q.m_system.m_units[i].m_class == qrhs.m_system.m_units[i].m_class ) )
        {
          if( qrhs.m_unit_powers.m_powers[i] != 0.0 )
          {
            if( q.m_unit_powers.m_powers[i] == 0.0 )
            {
              q.m_system.m_units[i] = qrhs.m_system.m_units[i];
              q.m_unit_powers.m_powers[i] = qrhs.m_unit_powers.m_powers[i];
            }
            else
            {
              q.m_value *= pow_ce_dint( qrhs.m_system.m_units[i].m_to_si / q.m_system.m_units[i].m_to_si , qrhs.m_unit_powers.m_powers[i] );
              q.m_unit_powers.m_powers[i] += qrhs.m_unit_powers.m_powers[i];
            }
          }
        }
        else
        {
          q.m_system.m_units[i] = unknown; // will popup a nan
          q.m_unit_powers.m_powers[i] = 0.0;
        }
      }
      return q;
    }

    /*
     * Raise a quantity to a power.
     * For instance, if q = 2 * m * (s^-2) then q^2 = 4 * (m^2) * (s^-4)
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator ^ ( const Quantity& qlhs , double power )
    {
      using onika::cuda::pow_ce_dint;
      Quantity q = qlhs;
      q.m_value = pow_ce_dint( q.m_value , power );
      for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
      {
        q.m_unit_powers.m_powers[i] *= power;
      }
      return q;
    }

    /*
     * When a scalar value mutliplies a UnitDefinition, it builds up a quantity
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator * ( double value , const UnitDefinition& U )
    {
      Quantity q = { value };
      if( U.m_class>=0 && U.m_class < NUMBER_OF_UNIT_CLASSES )
      {
        q.m_system.m_units[ U.m_class ] = U;
        q.m_unit_powers.m_powers[ U.m_class ] = 1.0;
      }
      // energy is dispatched to other units
      else if( U.m_class == ENERGY )
      {
        q.m_value *= U.m_to_si;
        q.m_system.m_units[ LENGTH ] = meter;
        q.m_unit_powers.m_powers[ LENGTH ] = 2;
        q.m_system.m_units[ MASS ] = kilogram;
        q.m_unit_powers.m_powers[ MASS ] = 1;
        q.m_system.m_units[ TIME ] = second;
        q.m_unit_powers.m_powers[ TIME ] = -2;        
      }
      else
      {
        q.m_value = undefined_value;
      }
      return q;
    }

    /*
     * When a UnitDefinition is raised to a power, it builds up a quantity with value=1.0
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator ^ ( const UnitDefinition& U , double power )
    {
      return ( 1.0 * U ) ^ power;
    }

    /*
     * Scalar multiplication of a quantity.
     * Exemple : 5.0 * ( 1.0 * m * (s^-2) ) = 5.0 * m * (s^-2)
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator * ( double value , const Quantity& qrhs )
    {
      Quantity q = qrhs;
      q.m_value *= value;
      return q;
    }

    /*
     * equivalent to building a quantity from 1.0 * UnitDefinition, and the nmultiplying the two quantities
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator * ( const Quantity& qlhs , const UnitDefinition& U )
    {
      return qlhs * ( 1.0 * U );
    }

    /*
     * qa / qb = qa * ( qb^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator / ( const Quantity& qlhs , const Quantity& qrhs )
    {
      return qlhs * ( qrhs ^ -1.0 );
    }

    /*
     * qa / U = qa * ( ( 1.0 * U ) ^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator / ( const Quantity& qlhs , const UnitDefinition& U )
    {
      return qlhs * ( ( 1.0 * U ) ^ -1.0 );
    }

    /*
     * x / U = x * ( ( 1.0 * U ) ^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator / ( double x , const UnitDefinition& U )
    {
      return x * ( ( 1.0 * U ) ^ -1.0 );
    }

    /*
     * x / q = x * ( q^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline constexpr Quantity operator / ( double x , const Quantity& qrhs )
    {
      return x * ( qrhs ^ -1.0 );
    }

    // pretty printing
    std::ostream& units_power_to_stream (std::ostream& out, const Quantity& q);
    std::ostream& operator << (std::ostream& out, const Quantity& q);

    // utulity functions to builds quantities from strings (i.e. YAML)
    Quantity make_quantity( double value , const std::string& units_and_powers );
    Quantity quantity_from_string(const std::string& s, bool& conversion_done);
    Quantity quantity_from_string(const std::string& s);
  }

}

/***************** YAML conversion *****************/

#include <yaml-cpp/yaml.h>
#include <sstream>
#include <onika/log.h>

namespace YAML
{
  template<> struct convert< onika::physics::UnitSystem >
  {
    static inline Node encode(const onika::physics::UnitSystem& us)
    {
      using namespace onika::physics;
      Node node(NodeType::Map);
      for(int i=0;i<=NUMBER_OF_UNIT_CLASSES;i++)
      {
        node[g_unit_class_str[i]] = us.m_units[i].m_name;
      }
      return node;
    }

    static inline bool decode(const Node& node, onika::physics::UnitSystem& us)
    {
      using namespace onika::physics;
      if( ! node.IsMap() )
      {
        onika::fatal_error() << "YAML node is not a map as expected when converting to onika::physics::UnitSystem" << std::endl << std::flush;
      }
      for(int i=0;i<=NUMBER_OF_UNIT_CLASSES;i++)
      {
        us.m_units[i] = unit_from_name( node[g_unit_class_str[i]].as<std::string>() );
      }
      return true;
    }
  };

  template<> struct convert< onika::physics::Quantity >
  {
    static inline Node encode(const onika::physics::Quantity& q)
    {
      std::ostringstream oss;
      oss << q;
      Node node;
      node = oss.str();
      return node;
    }

    static inline bool decode(const Node& node, onika::physics::Quantity& q)
    {
      if( node.IsScalar() ) // value and unit in the same string
      {
        bool conv_ok = false;
        q = onika::physics::quantity_from_string( node.as<std::string>() , conv_ok );
        return conv_ok;
      }
      else if( node.IsMap() )
      {
        q = onika::physics::make_quantity( node["value"].as<double>() , node["unity"].as<std::string>() );
        return true;
      }
      else
      {
        return false;
      }
    }

  };

}

// allow use of quantities without parsing strings
#define ONIKA_QUANTITY_DECL_SHORT_NAMES                                         \
  [[maybe_unused]] constexpr auto m      = ::onika::physics::meter;             \
  [[maybe_unused]] constexpr auto mm     = ::onika::physics::millimeter;        \
  [[maybe_unused]] constexpr auto um     = ::onika::physics::micron;            \
  [[maybe_unused]] constexpr auto nm     = ::onika::physics::nanometer;         \
  [[maybe_unused]] constexpr auto ang    = ::onika::physics::angstrom;          \
  [[maybe_unused]] constexpr auto kg     = ::onika::physics::kilogram;          \
  [[maybe_unused]] constexpr auto g      = ::onika::physics::gram;              \
  [[maybe_unused]] constexpr auto Da     = ::onika::physics::atomic_mass_unit;  \
  [[maybe_unused]] constexpr auto h      = ::onika::physics::hour;              \
  [[maybe_unused]] constexpr auto s      = ::onika::physics::second;            \
  [[maybe_unused]] constexpr auto us     = ::onika::physics::microsecond;       \
  [[maybe_unused]] constexpr auto ns     = ::onika::physics::nanosecond;        \
  [[maybe_unused]] constexpr auto ps     = ::onika::physics::picosecond;        \
  [[maybe_unused]] constexpr auto fs     = ::onika::physics::fetosecond;        \
  [[maybe_unused]] constexpr auto C      = ::onika::physics::coulomb;           \
  [[maybe_unused]] constexpr auto ec     = ::onika::physics::elementary_charge; \
  [[maybe_unused]] constexpr auto K      = ::onika::physics::kelvin;            \
  [[maybe_unused]] constexpr auto mol    = ::onika::physics::mol;               \
  [[maybe_unused]] constexpr auto cd     = ::onika::physics::candela;           \
  [[maybe_unused]] constexpr auto rad    = ::onika::physics::radian;            \
  [[maybe_unused]] constexpr auto degree = ::onika::physics::degree;            \
  [[maybe_unused]] constexpr auto J      = ::onika::physics::joule;             \
  [[maybe_unused]] constexpr auto eV     = ::onika::physics::electron_volt;     \
  [[maybe_unused]] constexpr auto cal    = ::onika::physics::calorie;           \
  [[maybe_unused]] constexpr auto kcal   = ::onika::physics::kcalorie

#define ONIKA_QUANTITY( __expr ) [&]() -> ::onika::physics::Quantity { ONIKA_QUANTITY_DECL_SHORT_NAMES; return ( __expr ); }()
#define ONIKA_CONST_QUANTITY( __expr ) []() -> ::onika::physics::Quantity { ONIKA_QUANTITY_DECL_SHORT_NAMES; return ( __expr ); }()

