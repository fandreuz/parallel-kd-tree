// Iterator used to iterate through a vector
template <class Pointer>
class vector_iterator : public vector_const_iterator<Pointer> {
public:
  explicit vector_iterator(Pointer ptr) : vector_const_iterator<Pointer>(ptr) {}

public:
  typedef std::random_access_iterator_tag iterator_category;
  typedef typename boost::intrusive::pointer_traits<Pointer>::element_type
      value_type;
  typedef typename boost::intrusive::pointer_traits<Pointer>::difference_type
      difference_type;
  typedef Pointer pointer;
  typedef value_type &reference;

  // Constructors
  vector_iterator() {}

  // Pointer like operators
  reference operator*() const { return *this->m_ptr; }

  value_type *operator->() const {
    return container_detail::to_raw_pointer(this->m_ptr);
  }

  reference operator[](difference_type off) const { return this->m_ptr[off]; }

  // Increment / Decrement
  vector_iterator &operator++() {
    ++this->m_ptr;
    return *this;
  }

  vector_iterator operator++(int) {
    pointer tmp = this->m_ptr;
    ++*this;
    return vector_iterator(tmp);
  }

  vector_iterator &operator--() {
    --this->m_ptr;
    return *this;
  }

  vector_iterator operator--(int) {
    vector_iterator tmp = *this;
    --*this;
    return vector_iterator(tmp);
  }

  // Arithmetic
  vector_iterator &operator+=(difference_type off) {
    this->m_ptr += off;
    return *this;
  }

  vector_iterator operator+(difference_type off) const {
    return vector_iterator(this->m_ptr + off);
  }

  friend vector_iterator operator+(difference_type off,
                                   const vector_iterator &right) {
    return vector_iterator(off + right.m_ptr);
  }

  vector_iterator &operator-=(difference_type off) {
    this->m_ptr -= off;
    return *this;
  }

  vector_iterator operator-(difference_type off) const {
    return vector_iterator(this->m_ptr - off);
  }

  difference_type operator-(const vector_const_iterator<Pointer> &right) const {
    return static_cast<const vector_const_iterator<Pointer> &>(*this) - right;
  }
};
