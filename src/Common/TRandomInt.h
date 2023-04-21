#ifndef TRANDOM_INT_H
#define TRANDOM_INT_H

#include <random>
#include <limits>

///Синглтон - генератор случайных беззнаковых целых с равномепным распределением по методу Mersenne Twister
class TRandomInt {
private:
  std::mt19937 RE;
	std::uniform_int_distribution<unsigned long> * URUL;

  ///Private constructor
	TRandomInt() {URUL = nullptr;}
  ~TRandomInt() {}  
	///Prevent copy-construction
	TRandomInt(const TRandomInt&);                 
  ///Prevent assignment
	TRandomInt& operator=(const TRandomInt&);      
public:
	///Получить экземпляр
  static TRandomInt& Instance()
  {
    static TRandomInt random_int;
    return random_int;
  }
	///Инициализация равномерного распределения
  void Initialize(unsigned long int s = 0)
  {
		if (s != 0) RE.seed(s); else RE.seed((unsigned int)time(nullptr));
		URUL = new std::uniform_int_distribution<unsigned long>(0, std::numeric_limits<unsigned long>::max());
	}
	///Получить случайный integer
  unsigned long operator () () {
		if (URUL == nullptr) Initialize();
    return (*URUL)(RE);
  }
};

///Глобальная функция для упрощения вызова
inline unsigned long RandomInt()
{
  TRandomInt * ri = &(TRandomInt::Instance());
  return (*ri)();
};

#endif
