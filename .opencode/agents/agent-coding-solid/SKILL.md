---
name: solid
description: Use this skill when writing code, implementing features, refactoring, planning architecture, designing systems, reviewing code, or debugging. This skill transforms junior-level code into senior-engineer quality software through SOLID principles, TDD, clean code practices, and professional software design.
---

# Solid Skills: Professional Software Engineering

You are now operating as a senior software engineer. Every line of code you write, every design decision you make, and every refactoring you perform must embody professional craftsmanship.

## When This Skill Applies

**ALWAYS use this skill when:**
- Writing ANY code (features, fixes, utilities)
- Refactoring existing code
- Planning or designing architecture
- Reviewing code quality
- Debugging issues
- Creating tests
- Making design decisions

## Core Philosophy

> "Code is to create products for users & customers. Testable, flexible, and maintainable code that serves the needs of the users is GOOD because it can be cost-effectively maintained by developers."

The goal of software: Enable developers to **discover, understand, add, change, remove, test, debug, deploy**, and **monitor** features efficiently.

## The Non-Negotiable Process

### 1. ALWAYS Start with Tests (TDD)

**Red-Green-Refactor is not optional:**

```
1. RED    - Write a failing test that describes the behavior
2. GREEN  - Write the SIMPLEST code to make it pass
3. REFACTOR - Clean up, remove duplication (Rule of Three)
```

**The Three Laws of TDD:**
1. You cannot write production code unless it makes a failing test pass
2. You cannot write more test code than is sufficient to fail
3. You cannot write more production code than is sufficient to pass

**Design happens during REFACTORING, not during coding.**

See: [references/tdd.md](references/tdd.md)

### 2. Apply SOLID Principles Rigorously

Every class, every module, every function:

| Principle | Question to Ask |
|-----------|-----------------|
| **S**RP - Single Responsibility | "Does this have ONE reason to change?" |
| **O**CP - Open/Closed | "Can I extend without modifying?" |
| **L**SP - Liskov Substitution | "Can subtypes replace base types safely?" |
| **I**SP - Interface Segregation | "Are clients forced to depend on unused methods?" |
| **D**IP - Dependency Inversion | "Do high-level modules depend on abstractions?" |

See: [references/solid-principles.md](references/solid-principles.md)

### 3. Write Clean, Human-Readable Code

**Naming (in order of priority):**
1. **Consistency** - Same concept = same name everywhere
2. **Understandability** - Domain language, not technical jargon
3. **Specificity** - Precise, not vague (avoid `data`, `info`, `manager`)
4. **Brevity** - Short but not cryptic
5. **Searchability** - Unique, greppable names

**Structure:**
- One level of indentation per method
- No `else` keyword when possible (early returns)
- When validating untrusted strings against an object/map, use `Object.hasOwn(...)` (or `Object.prototype.hasOwnProperty.call(...)`) — do not use the `in` operator, which matches prototype keys
- **ALWAYS wrap primitives in domain objects** - IDs, emails, money amounts, etc.
- First-class collections (wrap arrays in classes)
- One dot per line (Law of Demeter)
- Keep entities small (< 50 lines for classes, < 10 for methods)
- No more than two instance variables per class

**Value Objects are MANDATORY for:**
```typescript
// ALWAYS create value objects for:
class UserId { constructor(private readonly value: string) {} }
class Email { constructor(private readonly value: string) { /* validate */ } }
class Money { constructor(private readonly amount: number, private readonly currency: string) {} }
class OrderId { constructor(private readonly value: string) {} }

// NEVER use raw primitives for domain concepts:
// BAD: function createOrder(userId: string, email: string)
// GOOD: function createOrder(userId: UserId, email: Email)
```

See: [references/clean-code.md](references/clean-code.md)

### 4. Design with Responsibility in Mind

**Ask these questions for every class:**
1. "What pattern is this?" (Entity, Service, Repository, Factory, etc.)
2. "Is it doing too much?" (Check object calisthenics)

**Object Stereotypes:**
- **Information Holder** - Holds data, minimal behavior
- **Structurer** - Manages relationships between objects
- **Service Provider** - Performs work, stateless operations
- **Coordinator** - Orchestrates multiple services
- **Controller** - Makes decisions, delegates work
- **Interfacer** - Transforms data between systems

See: [references/object-design.md](references/object-design.md)

### 5. Manage Complexity Ruthlessly

**Essential complexity** = inherent to the problem domain
**Accidental complexity** = introduced by our solutions

**Detect complexity through:**
- Change amplification (small change = many files)
- Cognitive load (hard to understand)
- Unknown unknowns (surprises in behavior)

**Fight complexity with:**
- YAGNI - Don't build what you don't need NOW
- KISS - Simplest solution that works
- DRY - But only after Rule of Three (wait for 3 duplications)

See: [references/complexity.md](references/complexity.md)

### 6. Architect for Change

**Vertical Slicing:**
- Features as end-to-end slices
- Each feature self-contained

**Horizontal Decoupling:**
- Layers don't know about each other's internals
- Dependencies point inward (toward domain)

**The Dependency Rule:**
- Source code dependencies point toward high-level policies
- Infrastructure depends on domain, never reverse

See: [references/architecture.md](references/architecture.md)

## The Four Elements of Simple Design (XP)

In priority order:
1. **Runs all the tests** - Must work correctly
2. **Expresses intent** - Readable, reveals purpose
3. **No duplication** - DRY (but Rule of Three)
4. **Minimal** - Fewest classes, methods possible

## Code Smell Detection

**Stop and refactor when you see:**

| Smell | Solution |
|-------|----------|
| Long Method | Extract methods, compose method pattern |
| Large Class | Extract class, single responsibility |
| Long Parameter List | Introduce parameter object |
| Divergent Change | Split into focused classes |
| Shotgun Surgery | Move related code together |
| Feature Envy | Move method to the envied class |
| Data Clumps | Extract class for grouped data |
| Primitive Obsession | Wrap in value objects |
| Switch Statements | Replace with polymorphism |
| Parallel Inheritance | Merge hierarchies |
| Speculative Generality | YAGNI - remove unused abstractions |

See: [references/code-smells.md](references/code-smells.md)

## Design Patterns Awareness

**Creational:** Singleton, Factory, Builder, Prototype
**Structural:** Adapter, Bridge, Decorator, Composite, Proxy
**Behavioral:** Strategy, Observer, Template Method, Command

**Warning:** Don't force patterns. Let them emerge from refactoring.

See: [references/design-patterns.md](references/design-patterns.md)

## Testing Strategy

**Test Types (from inner to outer):**
1. **Unit Tests** - Single class/function, fast, isolated
2. **Integration Tests** - Multiple components together
3. **E2E/Acceptance Tests** - Full system, user perspective

**Arrange-Act-Assert Pattern:**
```typescript
// Arrange - Set up test state
const calculator = new Calculator();

// Act - Execute the behavior
const result = calculator.add(2, 3);

// Assert - Verify the outcome
expect(result).toBe(5);
```

**Test Naming:** Use concrete examples, not abstract statements
```typescript
// BAD: 'can add numbers'
// GOOD: 'when adding 2 + 3, returns 5'
```

See: [references/testing.md](references/testing.md)

## Behavioral Principles

- **Tell, Don't Ask** - Command objects, don't query and decide
- **Design by Contract** - Preconditions, postconditions, invariants
- **Hollywood Principle** - "Don't call us, we'll call you" (IoC)
- **Law of Demeter** - Only talk to immediate friends

## Pre-Code Checklist

Before writing ANY code, answer:

1. [ ] Do I understand the requirement? (Write acceptance criteria first)
2. [ ] What test will I write first?
3. [ ] What is the simplest solution?
4. [ ] What patterns might apply? (Don't force them)
5. [ ] Am I solving a real problem or a hypothetical one?

## During-Code Checklist

While coding, continuously ask:

1. [ ] Is this the simplest thing that could work?
2. [ ] Does this class have a single responsibility?
3. [ ] Am I depending on abstractions or concretions?
4. [ ] Can I name this more clearly?
5. [ ] Is there duplication I should extract? (Rule of Three)

## Post-Code Checklist

After the code works:

1. [ ] Do all tests pass?
2. [ ] Is there any dead code to remove?
3. [ ] Can I simplify any complex conditions?
4. [ ] Are names still accurate after changes?
5. [ ] Would a junior understand this in 6 months?

## Red Flags - Stop and Rethink

- Writing code without a test
- Class with more than 2 instance variables
- Method longer than 10 lines
- More than one level of indentation
- Using `else` when early return works
- Hardcoding values that should be configurable
- Creating abstractions before the third duplication
- Adding features "just in case"
- Depending on concrete implementations
- God classes that know everything

## Remember

> "A little bit of duplication is 10x better than the wrong abstraction."

> "Focus on WHAT needs to happen, not HOW it needs to happen."

> "Design principles become second nature through practice. Eventually, you won't think about SOLID - you'll just write SOLID code."

The journey: Code-first → Best-practice-first → Pattern-first → Responsibility-first → **Systems Thinking**

Your goal is to reach systems thinking - where principles are internalized and you focus on optimizing the entire development process.

# Software Architecture

## The Goal of Architecture

Enable the development team to:
1. **Add** features with minimal friction
2. **Change** existing features safely
3. **Remove** features cleanly
4. **Test** features in isolation
5. **Deploy** independently when possible

## Architectural Principles

### 1. Vertical Boundaries (Features/Slices)

Organize by **feature**, not by technical layer.

```
BAD: Layer-first
src/
  controllers/
    UserController.ts
    OrderController.ts
  services/
    UserService.ts
    OrderService.ts
  repositories/
    UserRepository.ts
    OrderRepository.ts

GOOD: Feature-first
src/
  users/
    UserController.ts
    UserService.ts
    UserRepository.ts
  orders/
    OrderController.ts
    OrderService.ts
    OrderRepository.ts
```

**Why:** Changes to "users" feature stay in `users/`. High cohesion within features.

### 2. Horizontal Boundaries (Layers)

Separate concerns into layers with clear dependencies.

```
┌──────────────────────────────────────┐
│           Presentation               │  UI, Controllers, CLI
├──────────────────────────────────────┤
│           Application                │  Use Cases, Orchestration
├──────────────────────────────────────┤
│             Domain                   │  Business Logic, Entities
├──────────────────────────────────────┤
│          Infrastructure              │  Database, APIs, External
└──────────────────────────────────────┘
```

### 3. The Dependency Rule

**Dependencies point INWARD.**

```
Infrastructure → Application → Domain
      ↓               ↓            ↓
   (outer)        (middle)      (inner)
```

- Inner layers know NOTHING about outer layers
- Domain has zero dependencies on infrastructure
- Use interfaces to invert dependencies

```typescript
// Domain defines the interface (inner)
interface UserRepository {
  save(user: User): Promise<void>;
  findById(id: UserId): Promise<User | null>;
}

// Infrastructure implements it (outer)
class PostgresUserRepository implements UserRepository {
  save(user: User): Promise<void> {
    // SQL here
  }
}

// Domain service uses the interface
class UserService {
  constructor(private repo: UserRepository) {} // Depends on abstraction
}
```

### 4. Contracts

Interfaces define boundaries between components.

```typescript
// The contract
interface PaymentGateway {
  charge(amount: Money, card: CardDetails): Promise<ChargeResult>;
  refund(chargeId: string): Promise<RefundResult>;
}

// Multiple implementations possible
class StripeGateway implements PaymentGateway { }
class PayPalGateway implements PaymentGateway { }
class MockGateway implements PaymentGateway { }  // For tests
```

### 5. Cross-Cutting Concerns

Concerns that span multiple features: logging, auth, validation, error handling.

**Options:**
- Middleware/interceptors
- Decorators
- Aspect-oriented approaches
- Base classes (use sparingly)

```typescript
// Middleware approach
class LoggingMiddleware {
  handle(request: Request, next: Handler): Response {
    console.log(`Request: ${request.path}`);
    const response = next(request);
    console.log(`Response: ${response.status}`);
    return response;
  }
}
```

### 6. Conway's Law

> "Organizations design systems that mirror their communication structure."

**Implication:** Team structure affects architecture. Align both intentionally.

---

## Common Architectural Styles

### Layered Architecture

Traditional layers: Presentation → Business → Persistence

**Pros:** Simple, well-understood
**Cons:** Can become a "big ball of mud" without discipline

### Hexagonal Architecture (Ports & Adapters)

Domain at center, adapters around the edges.

```
        ┌─────────────────────┐
        │     HTTP Adapter    │
        └─────────┬───────────┘
                  │
┌─────────────────▼─────────────────┐
│              DOMAIN                │
│   ┌─────────────────────────┐     │
│   │      Business Logic      │     │
│   │      Use Cases           │     │
│   └─────────────────────────┘     │
└─────────────────┬─────────────────┘
                  │
        ┌─────────▼───────────┐
        │   Database Adapter   │
        └─────────────────────┘
```

**Ports:** Interfaces defined by the domain
**Adapters:** Implementations that connect to the outside world

### Clean Architecture

Similar to Hexagonal, with explicit layers:

1. **Entities** - Enterprise business rules
2. **Use Cases** - Application business rules
3. **Interface Adapters** - Controllers, Presenters, Gateways
4. **Frameworks & Drivers** - Web, DB, External interfaces

---

## Feature-Driven Structure (Frontend)

```
src/
  features/
    auth/
      components/
        LoginForm.tsx
        SignupForm.tsx
      hooks/
        useAuth.ts
      services/
        authService.ts
      types/
        auth.types.ts
      index.ts  # Public API
    checkout/
      components/
      hooks/
      services/
      types/
      index.ts
  shared/
    components/  # Truly shared UI
    hooks/       # Truly shared hooks
    utils/       # Truly shared utilities
```

---

## Feature-Driven Structure (Backend)

```
src/
  modules/
    users/
      domain/
        User.ts
        UserRepository.ts  # Interface
      application/
        CreateUser.ts      # Use case
        GetUser.ts         # Use case
      infrastructure/
        PostgresUserRepo.ts
      presentation/
        UserController.ts
        UserDTO.ts
    orders/
      domain/
      application/
      infrastructure/
      presentation/
  shared/
    domain/        # Shared value objects
    infrastructure/ # Shared infra utilities
```

---

## The Walking Skeleton

Start with a minimal end-to-end slice:

1. **Thinnest possible feature** that touches all layers
2. **Deployable** from day one
3. **Proves the architecture** works

Example walking skeleton for e-commerce:
- User can view ONE product (hardcoded)
- User can add it to cart
- User can "checkout" (just logs)

From there, flesh out each feature fully.

---

## Testing Architecture

```
┌────────────────────────────────────────────┐
│            E2E / Acceptance Tests          │  Few, slow, high confidence
├────────────────────────────────────────────┤
│            Integration Tests               │  Some, medium speed
├────────────────────────────────────────────┤
│              Unit Tests                    │  Many, fast, isolated
└────────────────────────────────────────────┘
```

**Test by layer:**
- **Domain:** Unit tests (most tests here)
- **Application:** Integration tests with mocked infra
- **Infrastructure:** Integration tests with real dependencies
- **E2E:** Critical paths only

---

## Architecture Decision Records (ADRs)

Document significant decisions:

```markdown
# ADR 001: Use PostgreSQL for persistence

## Status
Accepted

## Context
We need a database. Options: PostgreSQL, MongoDB, MySQL

## Decision
PostgreSQL for:
- ACID compliance
- Team familiarity
- JSON support for flexibility

## Consequences
- Need PostgreSQL expertise
- Schema migrations required
- Excellent query capabilities
```

---

## Red Flags in Architecture

- **Circular dependencies** between modules
- **Domain depending on infrastructure**
- **Framework code in business logic**
- **No clear boundaries** between features
- **Shared mutable state** across modules
- **"Util" or "Common" packages** that grow forever
- **Database schema driving domain model**

# Clean Code Practices

## What is Clean Code?

Code that is:
- **Easy to understand** - reveals intent clearly
- **Easy to change** - modifications are localized
- **Easy to test** - dependencies are injectable
- **Simple** - no unnecessary complexity

## The Human-Centered Approach

Code has THREE consumers:
1. **Users** - get their needs met
2. **Customers** - make or save money
3. **Developers** - must maintain it

Design for all three, but remember: **developers read code 10x more than they write it.**

## Naming Principles

### 1. Consistency & Uniqueness (HIGHEST PRIORITY)
Same concept = same name everywhere. One name per concept.

```typescript
// BAD: Inconsistent names for same concept
getUserById(id)
fetchCustomerById(id)
retrieveClientById(id)

// GOOD: Consistent
getUser(id)
getOrder(id)
getProduct(id)
```

### 2. Understandability
Use domain language, not technical jargon.

```typescript
// BAD: Technical
const arr = users.filter(u => u.isActive);

// GOOD: Domain language
const activeCustomers = users.filter(user => user.isActive);
```

### 3. Specificity
Avoid vague names: `data`, `info`, `manager`, `handler`, `processor`, `utils`

```typescript
// BAD: Vague
class DataManager { }
function processInfo(data) { }

// GOOD: Specific
class OrderRepository { }
function validatePayment(payment) { }
```

### 4. Brevity (but not at cost of clarity)
Short names are good only if meaning is preserved.

```typescript
// BAD: Too cryptic
const usrLst = getUsrs();

// BAD: Unnecessarily long
const listOfAllActiveUsersInTheSystem = getActiveUsers();

// GOOD: Brief but clear
const activeUsers = getActiveUsers();
```

### 5. Searchability
Names should be unique enough to grep/search.

```typescript
// BAD: Common word, hard to search
const data = fetch();

// GOOD: Unique, searchable
const orderSummary = fetchOrderSummary();
```

### 6. Pronounceability
You should be able to say it in conversation.

```typescript
// BAD
const genymdhms = generateYearMonthDayHourMinuteSecond();

// GOOD
const timestamp = generateTimestamp();
```

### 7. Austerity
Avoid unnecessary filler words.

```typescript
// BAD: Redundant
const userData = user; // 'Data' adds nothing
class UserClass { }    // 'Class' adds nothing

// GOOD
const user = user;
class User { }
```

---

## Object Calisthenics (9 Rules)

Exercises to improve OO design. Follow strictly during practice, relax slightly in production.

### 1. One Level of Indentation per Method

```typescript
// BAD: Multiple levels
function process(orders: Order[]) {
  for (const order of orders) {
    if (order.isValid()) {
      for (const item of order.items) {
        if (item.inStock) {
          // process...
        }
      }
    }
  }
}

// GOOD: Extract methods
function process(orders: Order[]) {
  orders.filter(o => o.isValid()).forEach(processOrder);
}

function processOrder(order: Order) {
  order.items.filter(i => i.inStock).forEach(processItem);
}
```

### 2. Don't Use the ELSE Keyword

Use early returns, guard clauses, or polymorphism.

```typescript
// BAD: else
function getDiscount(user: User): number {
  if (user.isPremium) {
    return 20;
  } else {
    return 0;
  }
}

// GOOD: Early return
function getDiscount(user: User): number {
  if (user.isPremium) return 20;
  return 0;
}
```

### 3. Wrap All Primitives and Strings

Primitives should be wrapped in domain objects when they have meaning.

```typescript
// BAD: Primitive obsession
function createUser(email: string, age: number) { }

// GOOD: Value objects
class Email {
  constructor(private value: string) {
    if (!this.isValid(value)) throw new InvalidEmail();
  }
  private isValid(email: string): boolean { ... }
}

class Age {
  constructor(private value: number) {
    if (value < 0 || value > 150) throw new InvalidAge();
  }
}

function createUser(email: Email, age: Age) { }
```

### 4. First-Class Collections

Any class with a collection should have no other instance variables.

```typescript
// BAD: Collection mixed with other state
class Order {
  items: OrderItem[] = [];
  customerId: string;
  total: number;
}

// GOOD: Collection is its own class
class OrderItems {
  constructor(private items: OrderItem[] = []) {}

  add(item: OrderItem): void { ... }
  total(): Money { ... }
  isEmpty(): boolean { ... }
}

class Order {
  constructor(
    private items: OrderItems,
    private customerId: CustomerId
  ) {}
}
```

### 5. One Dot per Line (Law of Demeter)

Don't chain through object graphs.

```typescript
// BAD: Train wreck
const city = order.customer.address.city;

// GOOD: Tell, don't ask
const city = order.getShippingCity();
```

### 6. Don't Abbreviate

If a name is too long to type, the class is doing too much.

```typescript
// BAD
const custRepo = new CustRepo();
const ord = new Ord();

// GOOD
const customerRepository = new CustomerRepository();
const order = new Order();
```

### 7. Keep All Entities Small

- Classes: < 50 lines
- Methods: < 10 lines
- Files: < 100 lines

If larger, it's probably doing too much. Split it.

### 8. No Classes with More Than Two Instance Variables

Forces small, focused classes.

```typescript
// BAD: Too many variables
class Order {
  id: string;
  customerId: string;
  items: Item[];
  total: number;
  status: string;
}

// GOOD: Composed of smaller objects
class Order {
  constructor(
    private id: OrderId,
    private details: OrderDetails
  ) {}
}

class OrderDetails {
  constructor(
    private customer: Customer,
    private lineItems: LineItems
  ) {}
}
```

### 9. No Getters/Setters/Properties

Objects should have behavior, not just data. Tell objects what to do.

```typescript
// BAD: Data bag with getters
class Account {
  getBalance(): number { return this.balance; }
  setBalance(value: number) { this.balance = value; }
}

// Caller does the work
if (account.getBalance() >= amount) {
  account.setBalance(account.getBalance() - amount);
}

// GOOD: Behavior-rich object
class Account {
  withdraw(amount: Money): WithdrawResult {
    if (!this.canWithdraw(amount)) {
      return WithdrawResult.insufficientFunds();
    }
    this.balance = this.balance.subtract(amount);
    return WithdrawResult.success();
  }
}

// Caller tells, object decides
const result = account.withdraw(amount);
```

---

## Comments

### When to Write Comments

**Only write comments to explain WHY, not WHAT or HOW.**

Code explains what and how. Comments explain business reasons, non-obvious decisions, or warnings.

```typescript
// BAD: Explains what (redundant)
// Add 1 to counter
counter++;

// GOOD: Explains why
// Compensate for 0-based indexing in legacy API
counter++;
```

### Prefer Self-Documenting Code

Instead of commenting, rename to make intent clear.

```typescript
// BAD: Comment needed
// Check if user can access premium features
if (user.subscriptionLevel >= 2 && !user.isBanned) { }

// GOOD: Self-documenting
if (user.canAccessPremiumFeatures()) { }
```

---

## Formatting

### Vertical Spacing
- Related code together
- Blank lines between concepts
- Most important/public at top

### Horizontal Spacing
- Consistent indentation
- Space around operators
- Max line length ~80-120 characters

### Storytelling
Code should read top-to-bottom like a story. High-level at top, details below.

```typescript
class OrderProcessor {
  // Public API first
  process(order: Order): ProcessResult {
    this.validate(order);
    this.calculateTotals(order);
    return this.save(order);
  }

  // Supporting methods below, in order of appearance
  private validate(order: Order): void { ... }
  private calculateTotals(order: Order): void { ... }
  private save(order: Order): ProcessResult { ... }
}
```

# Code Smells & Anti-Patterns

## What Are Code Smells?

Indicators that something MAY be wrong. Not bugs, but design problems that make code hard to understand, change, or test.

## The Five Categories

### 1. Bloaters
Code that has grown too large.

| Smell | Symptom | Refactoring |
|-------|---------|-------------|
| **Long Method** | > 10 lines | Extract Method |
| **Large Class** | > 50 lines, multiple responsibilities | Extract Class |
| **Long Parameter List** | > 3 parameters | Introduce Parameter Object |
| **Data Clumps** | Same group of variables appear together | Extract Class |
| **Primitive Obsession** | Primitives instead of small objects | Wrap in Value Object |

### 2. Object-Orientation Abusers
Misuse of OO principles.

| Smell | Symptom | Refactoring |
|-------|---------|-------------|
| **Switch Statements** | Type checking, large switch/if-else | Replace with Polymorphism |
| **Parallel Inheritance** | Adding subclass requires adding another | Merge Hierarchies |
| **Refused Bequest** | Subclass doesn't use parent methods | Replace Inheritance with Delegation |
| **Alternative Classes** | Different interfaces, same concept | Rename, Extract Superclass |

### 3. Change Preventers
Code that makes changes difficult.

| Smell | Symptom | Refactoring |
|-------|---------|-------------|
| **Divergent Change** | One class changed for many reasons | Extract Class (SRP) |
| **Shotgun Surgery** | One change touches many classes | Move Method/Field together |
| **Parallel Inheritance** | (see above) | Merge Hierarchies |

### 4. Dispensables
Code that can be removed.

| Smell | Symptom | Refactoring |
|-------|---------|-------------|
| **Comments** | Explaining bad code | Rename, Extract Method |
| **Duplicate Code** | Copy-paste | Extract Method, Pull Up Method |
| **Dead Code** | Unreachable code | Delete |
| **Speculative Generality** | "Just in case" code | Delete (YAGNI) |
| **Lazy Class** | Class that does almost nothing | Inline Class |

### 5. Couplers
Excessive coupling between classes.

| Smell | Symptom | Refactoring |
|-------|---------|-------------|
| **Feature Envy** | Method uses another class's data extensively | Move Method |
| **Inappropriate Intimacy** | Classes know too much about each other | Move Method, Extract Class |
| **Message Chains** | `a.getB().getC().getD()` | Hide Delegate |
| **Middle Man** | Class only delegates | Inline Class |

---

## The Seven Most Common Code Smells

### 1. Long Method

**Symptom:** Method > 10 lines, doing multiple things.

```typescript
// SMELL
function processOrder(order: Order) {
  // Validate
  if (!order.items.length) throw new Error('Empty');
  if (!order.customer) throw new Error('No customer');

  // Calculate
  let total = 0;
  for (const item of order.items) {
    total += item.price * item.quantity;
    if (item.discount) {
      total -= item.discount;
    }
  }

  // Apply tax
  const taxRate = getTaxRate(order.customer.state);
  total = total * (1 + taxRate);

  // Save
  db.orders.insert({ ...order, total });

  // Notify
  emailService.send(order.customer.email, 'Order confirmed');
}

// REFACTORED
function processOrder(order: Order) {
  validateOrder(order);
  const total = calculateTotal(order);
  saveOrder(order, total);
  notifyCustomer(order);
}
```

### 2. Large Class

**Symptom:** Class with many responsibilities, > 50 lines.

```typescript
// SMELL: God class
class User {
  // User data
  name: string;
  email: string;

  // Authentication
  login() { }
  logout() { }
  resetPassword() { }

  // Preferences
  setTheme() { }
  setLanguage() { }

  // Notifications
  sendEmail() { }
  sendSMS() { }

  // Billing
  charge() { }
  refund() { }
}

// REFACTORED: Separate classes
class User { name: string; email: string; }
class AuthService { login(); logout(); resetPassword(); }
class UserPreferences { setTheme(); setLanguage(); }
class NotificationService { sendEmail(); sendSMS(); }
class BillingService { charge(); refund(); }
```

### 3. Feature Envy

**Symptom:** Method uses another class's data more than its own.

```typescript
// SMELL: Order envies Customer
class Order {
  calculateShipping(customer: Customer): number {
    if (customer.country === 'US') {
      if (customer.state === 'CA') return 10;
      return 15;
    }
    return 25;
  }
}

// REFACTORED: Move to Customer
class Customer {
  getShippingCost(): number {
    if (this.country === 'US') {
      if (this.state === 'CA') return 10;
      return 15;
    }
    return 25;
  }
}

class Order {
  calculateShipping(): number {
    return this.customer.getShippingCost();
  }
}
```

### 4. Primitive Obsession

**Symptom:** Using primitives for domain concepts.

```typescript
// SMELL
function createUser(email: string, age: number, zipCode: string) {
  // No validation, easy to pass wrong values
  if (!email.includes('@')) throw new Error();
  if (age < 0) throw new Error();
}

// REFACTORED: Value objects
class Email {
  constructor(private value: string) {
    if (!value.includes('@')) throw new InvalidEmail();
  }
}

class Age {
  constructor(private value: number) {
    if (value < 0 || value > 150) throw new InvalidAge();
  }
}

function createUser(email: Email, age: Age, address: Address) {
  // Type system prevents invalid data
}
```

### 5. Switch Statements

**Symptom:** Switching on type, repeated across codebase.

```typescript
// SMELL
function getArea(shape: Shape): number {
  switch (shape.type) {
    case 'circle': return Math.PI * shape.radius ** 2;
    case 'rectangle': return shape.width * shape.height;
    case 'triangle': return 0.5 * shape.base * shape.height;
  }
}

function getPerimeter(shape: Shape): number {
  switch (shape.type) { // Same switch again!
    case 'circle': return 2 * Math.PI * shape.radius;
    // ...
  }
}

// REFACTORED: Polymorphism
interface Shape {
  getArea(): number;
  getPerimeter(): number;
}

class Circle implements Shape {
  constructor(private radius: number) {}
  getArea(): number { return Math.PI * this.radius ** 2; }
  getPerimeter(): number { return 2 * Math.PI * this.radius; }
}
```

### 6. Inappropriate Intimacy

**Symptom:** Classes know too much about each other's internals.

```typescript
// SMELL
class Order {
  process() {
    const inventory = new Inventory();
    // Reaching into inventory's internals
    for (const item of this.items) {
      const stock = inventory.stockLevels[item.sku];
      if (stock.quantity < item.quantity) {
        throw new Error('Out of stock');
      }
      inventory.stockLevels[item.sku].quantity -= item.quantity;
    }
  }
}

// REFACTORED: Tell, don't ask
class Inventory {
  reserve(items: OrderItem[]): ReserveResult {
    // Inventory manages its own state
    for (const item of items) {
      if (!this.canReserve(item)) {
        return ReserveResult.outOfStock(item);
      }
    }
    this.deductStock(items);
    return ReserveResult.success();
  }
}

class Order {
  process(inventory: Inventory) {
    const result = inventory.reserve(this.items);
    if (!result.isSuccess()) {
      throw new OutOfStockError(result.failedItem);
    }
  }
}
```

### 7. Speculative Generality

**Symptom:** "Just in case" abstractions that aren't used.

```typescript
// SMELL: Over-engineered for hypothetical needs
interface PaymentProcessor {
  process(): void;
  rollback(): void;
  audit(): void;
  generateReport(): void;
  scheduleRecurring(): void;
}

class StripeProcessor implements PaymentProcessor {
  process() { /* actual code */ }
  rollback() { throw new Error('Not implemented'); }
  audit() { throw new Error('Not implemented'); }
  generateReport() { throw new Error('Not implemented'); }
  scheduleRecurring() { throw new Error('Not implemented'); }
}

// REFACTORED: YAGNI
interface PaymentProcessor {
  process(): void;
}

class StripeProcessor implements PaymentProcessor {
  process() { /* actual code */ }
}
// Add other methods when actually needed
```

---

## Prevention Strategies

1. **Follow Object Calisthenics** - Rules prevent most smells
2. **Practice TDD** - Tests reveal design problems early
3. **Review in pairs** - Fresh eyes catch smells
4. **Refactor continuously** - Don't let smells accumulate
5. **Apply SOLID** - Prevents structural smells
6. **Use static analysis** - Tools catch common issues

---

## When You Find a Smell

1. **Confirm it's a problem** - Not all smells need fixing
2. **Ensure test coverage** - Before refactoring
3. **Refactor in small steps** - Keep tests passing
4. **Commit frequently** - Easy to revert if needed

# Managing Complexity

## The Two Types of Complexity

### Essential Complexity
Inherent to the problem domain. Cannot be removed, only managed.
- Business rules
- Domain logic
- User requirements

### Accidental Complexity
Introduced by our solutions. CAN and SHOULD be minimized.
- Poor abstractions
- Unnecessary indirection
- Framework ceremony
- Technical debt

**Goal: Minimize accidental complexity while clearly expressing essential complexity.**

---

## Detecting Complexity

### 1. Change Amplification
Small changes require touching many files.

**Symptom:** "To add this field, I need to update 15 files."

**Cause:** Scattered responsibilities, poor abstraction boundaries.

### 2. Cognitive Load
Code is hard to understand, requires holding too much in memory.

**Symptom:** "I need to understand 10 other classes to understand this one."

**Cause:** Tight coupling, hidden dependencies, unclear naming.

### 3. Unknown Unknowns
Behavior is surprising, side effects are hidden.

**Symptom:** "I changed this, and something completely unrelated broke."

**Cause:** Global state, hidden dependencies, implicit contracts.

---

## The XP Values for Fighting Complexity

From Extreme Programming:

### 1. Communication
Code should communicate clearly. Names, structure, tests all contribute.

### 2. Simplicity
Do the simplest thing that could possibly work.

### 3. Feedback
Fast feedback loops catch complexity early. TDD, CI, code review.

### 4. Courage
Refactor aggressively. Don't let complexity accumulate.

### 5. Respect
Respect future readers (including yourself). Write for humans first.

---

## KISS - Keep It Simple, Silly

> "The simplest solution that works is usually the best."

### How to Apply:
1. Start with the obvious solution
2. Only add complexity when REQUIRED
3. Prefer boring, well-understood approaches
4. Question every abstraction

```typescript
// Over-engineered
class UserServiceFactoryProvider {
  private static instance: UserServiceFactoryProvider;

  static getInstance(): UserServiceFactoryProvider { ... }
  createFactory(): UserServiceFactory { ... }
}

// KISS
class UserService {
  getUser(id: string): User { ... }
}
```

---

## YAGNI - You Aren't Gonna Need It

> "Don't build features until they're actually needed."

### Warning Signs:
- "We might need this later"
- "It would be nice to have"
- "Just in case"
- "For future extensibility"

### The Cost of YAGNI Violations:
1. **Development time** - Building unused features
2. **Maintenance burden** - Code that must be maintained
3. **Cognitive load** - More to understand
4. **Wrong abstraction** - Guessing future needs incorrectly

```typescript
// YAGNI violation: Building for hypothetical needs
class User {
  // "We might need these someday"
  middleName?: string;
  secondaryEmail?: string;
  faxNumber?: string;
  linkedinProfile?: string;
  twitterHandle?: string;
}

// YAGNI: Only what's needed NOW
class User {
  name: string;
  email: Email;
}
```

---

## DRY - Don't Repeat Yourself (with The Rule of Three)

> "Every piece of knowledge should have a single, unambiguous representation."

### BUT: The Rule of Three

**Don't extract duplication until you see it THREE times.**

Why? The wrong abstraction is worse than duplication.

```
Duplication #1 → Leave it
Duplication #2 → Note it, leave it
Duplication #3 → NOW extract it
```

### Example:
```typescript
// First time - leave it
function processUserOrder(order) {
  validate(order);
  calculateTax(order);
  save(order);
}

// Second time - note the similarity, but leave it
function processGuestOrder(order) {
  validate(order);
  calculateTax(order);
  save(order);
  sendGuestEmail(order);
}

// Third time - NOW extract
function processCorporateOrder(order) {
  validate(order);
  calculateTax(order);
  save(order);
  applyCorporateDiscount(order);
}

// After three, extract the common parts
function processOrder(order: Order, postProcessing: (o: Order) => void) {
  validate(order);
  calculateTax(order);
  save(order);
  postProcessing(order);
}
```

---

## Separation of Concerns

> "Each module should address a single concern."

### Concerns to Separate:
- **Business logic** vs **Infrastructure**
- **What** (policy) vs **How** (mechanism)
- **Input** vs **Processing** vs **Output**
- **Data** vs **Behavior**

### Example:
```typescript
// BAD: Mixed concerns
class OrderProcessor {
  process(order: Order) {
    // Validation
    if (!order.items.length) throw new Error('Empty');

    // Business logic
    let total = 0;
    for (const item of order.items) {
      total += item.price * item.quantity;
    }

    // Persistence
    const db = new Database();
    db.query(`INSERT INTO orders...`);

    // Notification
    const email = new EmailClient();
    email.send(order.customer.email, 'Order confirmed');
  }
}

// GOOD: Separated concerns
class OrderProcessor {
  constructor(
    private validator: OrderValidator,
    private calculator: OrderCalculator,
    private repository: OrderRepository,
    private notifier: OrderNotifier
  ) {}

  process(order: Order): ProcessResult {
    this.validator.validate(order);
    const total = this.calculator.calculateTotal(order);
    const savedOrder = this.repository.save(order);
    this.notifier.notifyConfirmation(savedOrder);
    return ProcessResult.success(savedOrder);
  }
}
```

---

## Managing Technical Debt

### Types of Technical Debt:
1. **Deliberate** - Conscious trade-off for speed
2. **Accidental** - Mistakes, lack of knowledge
3. **Bit rot** - Code degrades over time

### The Boy Scout Rule:
> "Leave the code better than you found it."

Every time you touch code:
- Improve one small thing
- Fix one naming issue
- Extract one method
- Add one missing test

### When to Pay Down Debt:
- When it's in your path (you're already there)
- When it's blocking new features
- When it's causing bugs
- During dedicated refactoring time

### When NOT to Refactor:
- Code that works and won't change
- Code being replaced soon
- When you don't have tests

---

## The Four Elements of Simple Design

In priority order (from XP):

1. **Runs all the tests**
   - If it doesn't work, nothing else matters

2. **Expresses intent**
   - Clear names, obvious structure
   - Code tells the story

3. **No duplication**
   - DRY (but Rule of Three)
   - Single source of truth

4. **Minimal**
   - Fewest classes and methods possible
   - Remove anything unnecessary

If these four are true, the design is simple enough.

# Design Patterns

## What Are Design Patterns?

Reusable solutions to common design problems. A shared vocabulary for discussing design.

## WARNING: Don't Force Patterns

> "Let patterns emerge from refactoring, don't force them upfront."

Patterns should solve problems you HAVE, not problems you MIGHT have.

## When to Use Patterns

1. **You recognize the problem** - You've seen it before
2. **The pattern fits** - Not forcing it
3. **It simplifies** - Doesn't add unnecessary complexity
4. **Team understands it** - Shared knowledge

---

## Creational Patterns

### Singleton

**Purpose:** Ensure only one instance exists.

**When to use:** Global configuration, connection pools, logging.

**Warning:** Often overused. Consider dependency injection instead.

```typescript
class Logger {
  private static instance: Logger;

  private constructor() {}

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  log(message: string): void { ... }
}
```

### Factory

**Purpose:** Create objects without specifying exact class.

**When to use:** Object creation logic is complex, or varies by type.

```typescript
interface Notification {
  send(message: string): void;
}

class EmailNotification implements Notification { ... }
class SMSNotification implements Notification { ... }
class PushNotification implements Notification { ... }

class NotificationFactory {
  create(type: 'email' | 'sms' | 'push'): Notification {
    switch (type) {
      case 'email': return new EmailNotification();
      case 'sms': return new SMSNotification();
      case 'push': return new PushNotification();
    }
  }
}
```

### Builder

**Purpose:** Construct complex objects step by step.

**When to use:** Objects with many optional parameters, test data creation.

```typescript
class UserBuilder {
  private user: Partial<User> = {};

  withName(name: string): UserBuilder {
    this.user.name = name;
    return this;
  }

  withEmail(email: string): UserBuilder {
    this.user.email = email;
    return this;
  }

  withAge(age: number): UserBuilder {
    this.user.age = age;
    return this;
  }

  build(): User {
    return new User(
      this.user.name!,
      this.user.email!,
      this.user.age
    );
  }
}

// Usage
const user = new UserBuilder()
  .withName('Alice')
  .withEmail('alice@example.com')
  .build();
```

### Prototype

**Purpose:** Create new objects by cloning existing ones.

**When to use:** Object creation is expensive, or you need copies with slight variations.

```typescript
interface Prototype {
  clone(): Prototype;
}

class Document implements Prototype {
  constructor(
    public title: string,
    public content: string,
    public metadata: Metadata
  ) {}

  clone(): Document {
    return new Document(
      this.title,
      this.content,
      { ...this.metadata }
    );
  }
}
```

---

## Structural Patterns

### Adapter

**Purpose:** Make incompatible interfaces work together.

**When to use:** Integrating third-party libraries, legacy code.

```typescript
// Third-party library with different interface
class OldPaymentAPI {
  makePayment(cents: number): boolean { ... }
}

// Our interface
interface PaymentGateway {
  charge(amount: Money): ChargeResult;
}

// Adapter
class OldPaymentAdapter implements PaymentGateway {
  constructor(private oldAPI: OldPaymentAPI) {}

  charge(amount: Money): ChargeResult {
    const cents = amount.toCents();
    const success = this.oldAPI.makePayment(cents);
    return success ? ChargeResult.success() : ChargeResult.failed();
  }
}
```

### Decorator

**Purpose:** Add behavior to objects dynamically.

**When to use:** Adding features without modifying existing code.

```typescript
interface Notifier {
  send(message: string): void;
}

class EmailNotifier implements Notifier {
  send(message: string): void {
    console.log(`Email: ${message}`);
  }
}

// Decorators
class SMSDecorator implements Notifier {
  constructor(private wrapped: Notifier) {}

  send(message: string): void {
    this.wrapped.send(message);
    console.log(`SMS: ${message}`);
  }
}

class SlackDecorator implements Notifier {
  constructor(private wrapped: Notifier) {}

  send(message: string): void {
    this.wrapped.send(message);
    console.log(`Slack: ${message}`);
  }
}

// Usage - compose behaviors
const notifier = new SlackDecorator(
  new SMSDecorator(
    new EmailNotifier()
  )
);
notifier.send('Alert!'); // Sends to all three
```

### Proxy

**Purpose:** Control access to an object.

**When to use:** Lazy loading, access control, logging, caching.

```typescript
interface Image {
  display(): void;
}

class RealImage implements Image {
  constructor(private filename: string) {
    this.loadFromDisk(); // Expensive
  }

  private loadFromDisk(): void { ... }

  display(): void { ... }
}

// Lazy loading proxy
class ImageProxy implements Image {
  private realImage: RealImage | null = null;

  constructor(private filename: string) {}

  display(): void {
    if (!this.realImage) {
      this.realImage = new RealImage(this.filename);
    }
    this.realImage.display();
  }
}
```

### Composite

**Purpose:** Treat individual objects and compositions uniformly.

**When to use:** Tree structures, hierarchies (files/folders, UI components).

```typescript
interface Component {
  getPrice(): number;
}

class Product implements Component {
  constructor(private price: number) {}

  getPrice(): number {
    return this.price;
  }
}

class Box implements Component {
  private children: Component[] = [];

  add(component: Component): void {
    this.children.push(component);
  }

  getPrice(): number {
    return this.children.reduce(
      (sum, child) => sum + child.getPrice(),
      0
    );
  }
}

// Usage
const smallBox = new Box();
smallBox.add(new Product(10));
smallBox.add(new Product(20));

const bigBox = new Box();
bigBox.add(smallBox);
bigBox.add(new Product(50));

console.log(bigBox.getPrice()); // 80
```

---

## Behavioral Patterns

### Strategy

**Purpose:** Define a family of algorithms, make them interchangeable.

**When to use:** Multiple ways to do something, switchable at runtime.

```typescript
interface PricingStrategy {
  calculate(basePrice: number): number;
}

class RegularPricing implements PricingStrategy {
  calculate(basePrice: number): number {
    return basePrice;
  }
}

class PremiumDiscount implements PricingStrategy {
  calculate(basePrice: number): number {
    return basePrice * 0.8; // 20% off
  }
}

class BlackFriday implements PricingStrategy {
  calculate(basePrice: number): number {
    return basePrice * 0.5; // 50% off
  }
}

class ShoppingCart {
  constructor(private pricing: PricingStrategy) {}

  calculateTotal(items: Item[]): number {
    const base = items.reduce((sum, i) => sum + i.price, 0);
    return this.pricing.calculate(base);
  }
}
```

### Observer

**Purpose:** Notify multiple objects about state changes.

**When to use:** Event systems, pub/sub, reactive updates.

```typescript
interface Observer {
  update(event: Event): void;
}

class EventEmitter {
  private observers: Observer[] = [];

  subscribe(observer: Observer): void {
    this.observers.push(observer);
  }

  unsubscribe(observer: Observer): void {
    this.observers = this.observers.filter(o => o !== observer);
  }

  notify(event: Event): void {
    this.observers.forEach(o => o.update(event));
  }
}

// Usage
class OrderService extends EventEmitter {
  placeOrder(order: Order): void {
    // Process order...
    this.notify({ type: 'ORDER_PLACED', order });
  }
}

class EmailService implements Observer {
  update(event: Event): void {
    if (event.type === 'ORDER_PLACED') {
      this.sendConfirmation(event.order);
    }
  }
}
```

### Template Method

**Purpose:** Define algorithm skeleton, let subclasses override steps.

**When to use:** Common algorithm with varying steps.

```typescript
abstract class DataExporter {
  // Template method - defines the algorithm
  export(data: Data[]): void {
    this.validate(data);
    const formatted = this.format(data);
    this.write(formatted);
    this.notify();
  }

  // Common steps
  private validate(data: Data[]): void { ... }
  private notify(): void { ... }

  // Steps to override
  protected abstract format(data: Data[]): string;
  protected abstract write(content: string): void;
}

class CSVExporter extends DataExporter {
  protected format(data: Data[]): string {
    return data.map(d => d.toCSV()).join('\n');
  }

  protected write(content: string): void {
    fs.writeFileSync('export.csv', content);
  }
}

class JSONExporter extends DataExporter {
  protected format(data: Data[]): string {
    return JSON.stringify(data);
  }

  protected write(content: string): void {
    fs.writeFileSync('export.json', content);
  }
}
```

### Command

**Purpose:** Encapsulate a request as an object.

**When to use:** Undo/redo, queuing, logging actions.

```typescript
interface Command {
  execute(): void;
  undo(): void;
}

class AddItemCommand implements Command {
  constructor(
    private cart: Cart,
    private item: Item
  ) {}

  execute(): void {
    this.cart.add(this.item);
  }

  undo(): void {
    this.cart.remove(this.item);
  }
}

class CommandHistory {
  private history: Command[] = [];

  execute(command: Command): void {
    command.execute();
    this.history.push(command);
  }

  undo(): void {
    const command = this.history.pop();
    command?.undo();
  }
}
```

---

## Pattern Awareness

### The Four-Dimensional Lens

When analyzing new code/libraries, ask:

1. **What problem does it solve?** (Creational, Structural, Behavioral)
2. **What scope?** (Object-level, Class-level, System-level)
3. **When is it applied?** (Compile-time, Runtime)
4. **How coupled?** (Tight, Loose)

This helps recognize patterns even in unfamiliar code.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| **God Object** | Class does everything | Split by responsibility |
| **Spaghetti Code** | Tangled, no structure | Refactor to layers |
| **Golden Hammer** | Using one pattern for everything | Match pattern to problem |
| **Premature Optimization** | Optimizing before needed | YAGNI, profile first |
| **Copy-Paste Programming** | Duplication | Extract, Rule of Three |


# Object-Oriented Design

## Responsibility-Driven Design (RDD)

The key insight: **Objects are defined by their responsibilities, not their data.**

### Finding Objects

Start with:
1. **Nouns** in requirements → candidate objects
2. **Verbs** → candidate methods/behaviors
3. **Domain concepts** → value objects

### Finding Responsibilities

Each object should answer:
- What does this object **know**?
- What does this object **do**?
- What does this object **decide**?

### Object Stereotypes

Every class fits one (or maybe two) stereotypes:

| Stereotype | Purpose | Example |
|------------|---------|---------|
| **Information Holder** | Knows things, holds data | `User`, `Product`, `Address` |
| **Structurer** | Maintains relationships | `OrderItems`, `UserGroup` |
| **Service Provider** | Performs work | `PaymentProcessor`, `EmailSender` |
| **Coordinator** | Orchestrates workflow | `OrderFulfillmentService` |
| **Controller** | Makes decisions, delegates | `CheckoutController` |
| **Interfacer** | Transforms between systems | `UserAPIAdapter`, `DatabaseMapper` |

### The Two Questions

For every class, ask:
1. **"What pattern is this?"** - Which stereotype? Which design pattern?
2. **"Is it doing too much?"** - Check object calisthenics rules

If you can't answer clearly, the class needs refactoring.

---

## Tell, Don't Ask

**Command objects to do work. Don't interrogate them and do the work yourself.**

```typescript
// BAD: Asking, then doing
if (account.getBalance() >= amount) {
  account.setBalance(account.getBalance() - amount);
  // more logic here...
}

// GOOD: Telling
const result = account.withdraw(amount);
if (result.isSuccess()) {
  // ...
}
```

The object that has the data should have the behavior.

---

## Design by Contract (DbC)

Every method has:
- **Preconditions** - What must be true BEFORE calling
- **Postconditions** - What will be true AFTER calling
- **Invariants** - What is ALWAYS true about the object

```typescript
class BankAccount {
  private balance: Money;

  // INVARIANT: balance is never negative

  // PRECONDITION: amount > 0
  // POSTCONDITION: balance decreased by amount OR error returned
  withdraw(amount: Money): WithdrawResult {
    if (amount.isNegativeOrZero()) {
      return WithdrawResult.invalidAmount();
    }

    if (this.balance.isLessThan(amount)) {
      return WithdrawResult.insufficientFunds();
    }

    this.balance = this.balance.minus(amount);
    return WithdrawResult.success(this.balance);
  }
}
```

---

## Composition Over Inheritance

**Prefer composing objects over extending classes.**

### Why Inheritance is Problematic:
- Tight coupling between parent and child
- Fragile base class problem
- Difficult to change parent without breaking children
- Forces "is-a" relationship that may not fit

### When to Use Inheritance:
- True "is-a" relationship (rare)
- Framework requirements
- Template Method pattern (intentional)

### Prefer Composition:
```typescript
// BAD: Inheritance
class PremiumUser extends User {
  getDiscount(): number { return 20; }
}

// GOOD: Composition
class User {
  constructor(private discountPolicy: DiscountPolicy) {}

  getDiscount(): number {
    return this.discountPolicy.calculate();
  }
}

// Now discount behavior is pluggable
new User(new PremiumDiscount());
new User(new StandardDiscount());
new User(new NoDiscount());
```

---

## The Law of Demeter (Principle of Least Knowledge)

**Only talk to your immediate friends.**

A method should only call:
1. Methods on `this`
2. Methods on parameters
3. Methods on objects it creates
4. Methods on its direct components

```typescript
// BAD: Reaching through objects
order.getCustomer().getAddress().getCity();

// GOOD: Ask the immediate friend
order.getShippingCity();
```

This reduces coupling - changes to `Address` don't ripple through all callers.

---

## Encapsulation

**Hide internal details, expose behavior.**

### Levels of Encapsulation:
1. **Data** - private fields, no direct access
2. **Implementation** - how things work internally
3. **Type** - concrete class hidden behind interface
4. **Design** - architectural decisions hidden from clients

```typescript
// BAD: Exposed internals
class Order {
  public items: Item[] = [];
  public total: number = 0;
}

// Client can corrupt state
order.items.push(item);
order.total = -999; // Oops!

// GOOD: Encapsulated
class Order {
  private items: OrderItems;
  private total: Money;

  addItem(item: Item): void {
    this.items.add(item);
    this.recalculateTotal();
  }

  getTotal(): Money {
    return this.total; // Returns copy or immutable
  }
}
```

---

## Polymorphism

**Replace conditionals with types.**

```typescript
// BAD: Type checking
function calculateShipping(method: string, value: number): number {
  if (method === 'standard') return value < 50 ? 5 : 0;
  if (method === 'express') return 15;
  if (method === 'overnight') return 25;
  throw new Error('Unknown method');
}

// GOOD: Polymorphism
interface ShippingMethod {
  calculateCost(orderValue: number): number;
}

class StandardShipping implements ShippingMethod {
  calculateCost(orderValue: number): number {
    return orderValue < 50 ? 5 : 0;
  }
}

class ExpressShipping implements ShippingMethod {
  calculateCost(orderValue: number): number {
    return 15;
  }
}

// Usage - no conditionals
function calculateShipping(method: ShippingMethod, value: number): number {
  return method.calculateCost(value);
}
```

---

## Value Objects vs Entities

### Value Objects
- Defined by their attributes (no identity)
- Immutable
- Comparable by value
- Examples: `Money`, `Email`, `Address`, `DateRange`

```typescript
class Money {
  constructor(
    private readonly amount: number,
    private readonly currency: string
  ) {}

  equals(other: Money): boolean {
    return this.amount === other.amount &&
           this.currency === other.currency;
  }

  add(other: Money): Money {
    if (this.currency !== other.currency) {
      throw new CurrencyMismatch();
    }
    return new Money(this.amount + other.amount, this.currency);
  }
}
```

### Entities
- Have identity (survives attribute changes)
- Usually mutable (via methods)
- Comparable by identity
- Examples: `User`, `Order`, `Product`

```typescript
class User {
  constructor(
    private readonly id: UserId,
    private email: Email,
    private name: Name
  ) {}

  equals(other: User): boolean {
    return this.id.equals(other.id); // Identity comparison
  }

  changeEmail(newEmail: Email): void {
    this.email = newEmail; // Still same user
  }
}
```

---

## Aggregates

A cluster of objects treated as a single unit for data changes.

- One object is the **aggregate root** (entry point)
- External code only references the root
- Root enforces invariants for the entire cluster

```typescript
// Order is the aggregate root
class Order {
  private items: OrderItem[] = [];

  // All access through the root
  addItem(product: Product, quantity: number): void {
    const item = new OrderItem(product, quantity);
    this.items.push(item);
    this.validateTotal();
  }

  removeItem(itemId: ItemId): void {
    this.items = this.items.filter(i => !i.id.equals(itemId));
  }

  // Root enforces invariants
  private validateTotal(): void {
    if (this.calculateTotal().exceeds(MAX_ORDER_VALUE)) {
      throw new OrderTotalExceeded();
    }
  }
}

// BAD: Accessing items directly
order.items.push(new OrderItem(...)); // Bypasses validation!

// GOOD: Through the root
order.addItem(product, 2); // Validation happens
```

# SOLID Principles

## Overview

SOLID helps structure software to be flexible, maintainable, and testable. These principles reduce coupling and increase cohesion.

## S - Single Responsibility Principle (SRP)

> "A class should have one, and only one, reason to change."

### Problem It Solves
God objects that do everything - hard to test, hard to change, hard to understand.

### How to Apply
Each class handles ONE responsibility. If you find yourself saying "and" when describing what a class does, split it.

```typescript
// BAD: Multiple responsibilities
class Order {
  calculateTotal(): number { ... }
  saveToDatabase(): void { ... }    // Persistence
  generateInvoice(): string { ... } // Presentation
}

// GOOD: Single responsibility each
class Order {
  private items: OrderItem[] = [];

  addItem(item: OrderItem): void { ... }
  calculateTotal(): number { ... }
}

class OrderRepository {
  save(order: Order): Promise<void> { ... }
}

class InvoiceGenerator {
  generate(order: Order): Invoice { ... }
}
```

### Detection Questions
- Does this class have multiple reasons to change?
- Can I describe it without using "and"?
- Would different stakeholders request changes to different parts?

---

## O - Open/Closed Principle (OCP)

> "Software entities should be open for extension but closed for modification."

### Problem It Solves
Having to modify existing, tested code every time requirements change. Risk of breaking working features.

### How to Apply
Design abstractions that allow new behavior through new classes, not edits to existing ones.

```typescript
// BAD: Must modify to add new shipping
class ShippingCalculator {
  calculate(type: string, value: number): number {
    if (type === 'standard') return value < 50 ? 5 : 0;
    if (type === 'express') return 15;
    // Must add more ifs for new types!
  }
}

// GOOD: Open for extension
interface ShippingMethod {
  calculateCost(orderValue: number): number;
}

class StandardShipping implements ShippingMethod {
  calculateCost(orderValue: number): number {
    return orderValue < 50 ? 5 : 0;
  }
}

class ExpressShipping implements ShippingMethod {
  calculateCost(orderValue: number): number {
    return 15;
  }
}

// Add new shipping by creating new class, not modifying existing
class SameDayShipping implements ShippingMethod {
  calculateCost(orderValue: number): number {
    return 25;
  }
}
```

### Architectural Insight
OCP at architecture level means: **design your codebase so new features are added by adding code, not changing existing code.**

---

## L - Liskov Substitution Principle (LSP)

> "Subtypes must be substitutable for their base types without altering program correctness."

### Problem It Solves
Subclasses that break expectations, requiring type-checking and special cases.

### How to Apply
Subclasses must honor the contract of the parent. If the parent returns positive numbers, subclasses cannot return negatives.

```typescript
// BAD: Violates parent's contract
class DiscountPolicy {
  getDiscount(value: number): number {
    return 0; // Non-negative expected
  }
}

class WeirdDiscount extends DiscountPolicy {
  getDiscount(value: number): number {
    return -5; // Increases cost! Breaks expectations
  }
}

// GOOD: Enforces contract
class DiscountPolicy {
  constructor(private discount: number) {
    if (discount < 0) throw new Error("Discount must be non-negative");
  }

  getDiscount(): number {
    return this.discount;
  }
}
```

### Key Insight
This is why you can swap `InMemoryUserRepo` for `PostgresUserRepo` - they both honor the `UserRepo` interface contract.

---

## I - Interface Segregation Principle (ISP)

> "Clients should not be forced to depend on methods they do not use."

### Problem It Solves
Fat interfaces that force partial implementations, empty methods, or throws.

### How to Apply
Split large interfaces into smaller, cohesive ones. Clients depend only on what they need.

```typescript
// BAD: Fat interface
interface WarehouseDevice {
  printLabel(orderId: string): void;
  scanBarcode(): string;
  packageItem(orderId: string): void;
}

class BasicPrinter implements WarehouseDevice {
  printLabel(orderId: string): void { /* works */ }
  scanBarcode(): string { throw new Error("Not supported"); } // Forced!
  packageItem(orderId: string): void { throw new Error("Not supported"); }
}

// GOOD: Segregated interfaces
interface LabelPrinter {
  printLabel(orderId: string): void;
}

interface BarcodeScanner {
  scanBarcode(): string;
}

interface ItemPackager {
  packageItem(orderId: string): void;
}

class BasicPrinter implements LabelPrinter {
  printLabel(orderId: string): void { /* only what it does */ }
}
```

### Detection
If you see `throw new Error("Not implemented")` or empty method bodies, the interface is too fat.

---

## D - Dependency Inversion Principle (DIP)

> "High-level modules should not depend on low-level modules. Both should depend on abstractions."

### Problem It Solves
Tight coupling to specific implementations (databases, APIs, frameworks). Hard to test, hard to swap.

### How to Apply
Depend on interfaces, inject implementations.

```typescript
// BAD: Direct dependency on concrete class
class OrderService {
  private emailService = new SendGridEmailService(); // Locked in!

  confirmOrder(email: string): void {
    this.emailService.send(email, "Order confirmed");
  }
}

// GOOD: Depend on abstraction
interface EmailService {
  send(to: string, message: string): void;
}

class OrderService {
  constructor(private emailService: EmailService) {}

  confirmOrder(email: string): void {
    this.emailService.send(email, "Order confirmed");
  }
}

// Now can inject any implementation
new OrderService(new SendGridEmailService());
new OrderService(new SESEmailService());
new OrderService(new MockEmailService()); // For tests!
```

### The Dependency Rule
Source code dependencies should point **inward** toward high-level policies (domain logic), never toward low-level details (infrastructure).

```
Infrastructure → Application → Domain
      ↑              ↑            ↑
    (outer)       (middle)     (inner)

Dependencies flow: outer → inner
Never: inner → outer
```

---

## Applying SOLID at Architecture Level

These principles scale beyond classes:

| Principle | Architecture Application |
|-----------|--------------------------|
| SRP | Each bounded context has one responsibility |
| OCP | New features = new modules, not edits to existing |
| LSP | Microservices with same contract are substitutable |
| ISP | Thin interfaces between services |
| DIP | High-level business logic doesn't know about databases/frameworks |

---

## Quick Reference

| Principle | One-Liner | Red Flag |
|-----------|-----------|----------|
| SRP | One reason to change | "This class handles X and Y and Z" |
| OCP | Add, don't modify | `if/else` chains for types |
| LSP | Subtypes are substitutable | Type-checking in calling code |
| ISP | Small, focused interfaces | Empty method implementations |
| DIP | Depend on abstractions | `new ConcreteClass()` in business logic |

# Test-Driven Development

## The Core Loop

```
RED → GREEN → REFACTOR → RED → ...
```

### RED Phase
Write a failing test that describes the behavior you want. The test should:
- Use domain language, not technical jargon
- Describe WHAT, not HOW
- Be a concrete example, not an abstract statement

```typescript
// BAD: Abstract
it('can add numbers', () => { ... });

// GOOD: Concrete example
it('when adding 2 + 3, returns 5', () => { ... });
```

### GREEN Phase
Write the **simplest possible code** to make the test pass. Two strategies:

1. **Fake It** - Return a hardcoded value
   ```typescript
   add(a: number, b: number): number {
     return 5; // Simplest thing!
   }
   ```

2. **Obvious Implementation** - If you know the solution
   ```typescript
   add(a: number, b: number): number {
     return a + b;
   }
   ```

**Prefer Fake It** when learning or unsure. Let more tests drive the real implementation.

### REFACTOR Phase
This is where **design happens**. Look for:
- Duplication (but wait for Rule of Three)
- Long methods to extract
- Poor names to improve
- Complex conditions to simplify

## The Three Laws of TDD

1. **No production code** without a failing test
2. **No more test code** than sufficient to fail (compilation failures count)
3. **No more production code** than sufficient to pass the one failing test

## The Rule of Three

**Only extract duplication when you see it THREE times.**

Why? Wrong abstractions are worse than duplication. Wait for the pattern to emerge.

```typescript
// Duplication #1 - Leave it
// Duplication #2 - Note it, leave it
// Duplication #3 - NOW extract it
```

## Triangulation

Each new test "sculpts" the solution toward a general, robust implementation.

Think of **degrees of freedom** - like a car that needs forward/back, left/right, and rotation. Each test carves out one degree of freedom until the implementation handles all cases.

## Transformation Priority Premise

When going from RED to GREEN, prefer simpler transformations:

| Priority | Transformation |
|----------|----------------|
| 1 | {} → nil |
| 2 | nil → constant |
| 3 | constant → variable |
| 4 | unconditional → conditional |
| 5 | scalar → collection |
| 6 | statement → recursion |
| 7 | value → mutated value |

Higher priority = simpler. Avoid jumping to complex transformations too early.

## Arrange-Act-Assert

Structure every test:

```typescript
it('calculates total with discount', () => {
  // ARRANGE - Set up the world
  const order = new Order();
  order.addItem({ price: 100 });
  const discount = new PercentDiscount(10);

  // ACT - Execute the behavior
  const total = order.calculateTotal(discount);

  // ASSERT - Verify the outcome
  expect(total).toBe(90);
});
```

## Writing Tests Backwards

Sometimes it helps to write AAA in reverse:
1. Write the ASSERT first - what do you want to verify?
2. Write the ACT - what action produces that result?
3. Write the ARRANGE - what setup is needed?

## Test Naming Principles

- Use **behavior-driven names** with domain language
- Provide **concrete examples**, not abstract statements
- **One example per test** for easy debugging
- Avoid leaking implementation details

```typescript
// BAD: Technical, implementation-focused
it('should set the data property to 1', () => { ... });

// GOOD: Behavior-focused, domain language
it('should recognize "mom" as a palindrome', () => { ... });
```

## Classic vs Mockist TDD

**Classic (Detroit/Chicago) TDD:**
- Test with real dependencies
- Higher confidence, slower tests
- Best for: Pure functions, integration tests

**Mockist (London) TDD:**
- Mock external dependencies
- Faster tests, more isolated
- Best for: Classes with infrastructure dependencies

Start with Classic TDD to learn the technique. Add mocks when testing code with databases, APIs, etc.

## Common Mistakes

1. **Writing code before tests** - Violates the fundamental principle
2. **Writing too much test** - Just enough to fail
3. **Writing too much code** - Just enough to pass
4. **Skipping refactor** - This is where design lives
5. **Testing implementation** - Test behavior, not how it's done
6. **Abstract test names** - Use concrete examples
7. **Extracting too early** - Wait for Rule of Three

# Testing Strategy

## The Testing Pyramid

```
       /\
      /  \        E2E Tests (Few)
     /----\       - Full system
    /      \      - Slow, brittle
   /--------\
  /          \    Integration Tests (Some)
 /------------\   - Multiple components
/              \  - Medium speed
----------------
      Unit Tests (Many)
      - Single unit
      - Fast, isolated
```

## Test Types

### Unit Tests

Test ONE class or function in isolation.

**Characteristics:**
- Fast (milliseconds)
- No external dependencies (mocked)
- Most of your tests should be unit tests

```typescript
describe('Order', () => {
  it('calculates total correctly', () => {
    const order = new Order();
    order.addItem({ price: 100 });
    order.addItem({ price: 50 });

    expect(order.calculateTotal()).toBe(150);
  });
});
```

### Integration Tests

Test multiple components together.

**Characteristics:**
- Slower (may use real DB)
- Test boundaries between components
- Fewer than unit tests

```typescript
describe('OrderService Integration', () => {
  let db: Database;
  let service: OrderService;

  beforeAll(async () => {
    db = await Database.connect();
    service = new OrderService(new PostgresOrderRepo(db));
  });

  it('saves and retrieves an order', async () => {
    const order = Order.create({ customerId: '123' });
    await service.save(order);

    const retrieved = await service.findById(order.id);
    expect(retrieved).toEqual(order);
  });
});
```

### E2E / Acceptance Tests

Test the entire system from user perspective.

**Characteristics:**
- Slowest
- Most brittle (many moving parts)
- Test critical paths only

```typescript
describe('Checkout Flow', () => {
  it('user can complete purchase', async () => {
    await page.goto('/products');
    await page.click('[data-testid="add-to-cart"]');
    await page.click('[data-testid="checkout"]');
    await page.fill('[name="card"]', '4242424242424242');
    await page.click('[data-testid="pay"]');

    expect(await page.textContent('h1')).toBe('Order Confirmed');
  });
});
```

---

## Arrange-Act-Assert (AAA)

Structure EVERY test this way:

```typescript
it('applies discount to premium users', () => {
  // ARRANGE - Set up the test world
  const user = new User({ isPremium: true });
  const cart = new Cart(user);
  cart.addItem({ price: 100 });

  // ACT - Execute the behavior under test
  const total = cart.calculateTotal();

  // ASSERT - Verify the expected outcome
  expect(total).toBe(80); // 20% discount
});
```

### Writing AAA Backwards

Sometimes easier to write in reverse:

1. **Assert first** - What do you want to verify?
2. **Act** - What action produces that result?
3. **Arrange** - What setup is needed?

---

## Test Naming

### Bad: Abstract, Technical

```typescript
it('should work correctly')
it('handles the edge case')
it('sets the data property')
```

### Good: Concrete Examples, Domain Language

```typescript
it('calculates 20% discount for premium users')
it('returns error when cart is empty')
it('recognizes "racecar" as a palindrome')
```

### Format

```typescript
// Option 1: should + behavior
it('should apply tax based on shipping state')

// Option 2: when + then
it('when adding 2 + 3, then returns 5')

// Option 3: Given-When-Then (for complex scenarios)
describe('given a premium user', () => {
  describe('when they checkout', () => {
    it('then they receive 20% discount', () => { ... });
  });
});
```

---

## Test Doubles

### Dummy

Object passed but never used.

```typescript
const dummyLogger = {} as Logger;
new UserService(realRepo, dummyLogger);
```

### Stub

Returns predefined values.

```typescript
const stubRepo: UserRepo = {
  findById: () => Promise.resolve(new User({ name: 'Test' })),
  save: () => Promise.resolve(),
};
```

### Spy

Records how it was called.

```typescript
const emailSpy = {
  sentEmails: [] as string[],
  send(to: string, message: string) {
    this.sentEmails.push(to);
  }
};

// Later
expect(emailSpy.sentEmails).toContain('user@example.com');
```

### Mock

Verifies expected interactions.

```typescript
const mockRepo = jest.fn<UserRepo>();
mockRepo.save.mockResolvedValue(undefined);

// After test
expect(mockRepo.save).toHaveBeenCalledWith(expectedUser);
```

### Fake

Working implementation (simplified).

```typescript
class InMemoryUserRepo implements UserRepo {
  private users: Map<string, User> = new Map();

  async save(user: User): Promise<void> {
    this.users.set(user.id, user);
  }

  async findById(id: string): Promise<User | null> {
    return this.users.get(id) || null;
  }
}
```

---

## Testing Strategies by Layer

### Domain Layer (Most Tests)

- Unit tests with no mocks
- Test business rules, value objects, entities
- Fast, comprehensive

```typescript
describe('Money', () => {
  it('adds amounts with same currency', () => {
    const a = Money.dollars(10);
    const b = Money.dollars(20);
    expect(a.add(b).equals(Money.dollars(30))).toBe(true);
  });

  it('throws when adding different currencies', () => {
    const usd = Money.dollars(10);
    const eur = Money.euros(10);
    expect(() => usd.add(eur)).toThrow(CurrencyMismatch);
  });
});
```

### Application Layer

- Integration tests with mocked infrastructure
- Test use case orchestration

```typescript
describe('CreateOrderUseCase', () => {
  it('creates order and sends confirmation', async () => {
    const orderRepo = new InMemoryOrderRepo();
    const emailService = { send: jest.fn() };
    const useCase = new CreateOrderUseCase(orderRepo, emailService);

    await useCase.execute({ customerId: '123', items: [...] });

    expect(orderRepo.count()).toBe(1);
    expect(emailService.send).toHaveBeenCalled();
  });
});
```

### Infrastructure Layer

- Integration tests with real dependencies
- Test database, API integrations

```typescript
describe('PostgresOrderRepo', () => {
  let repo: PostgresOrderRepo;

  beforeAll(async () => {
    repo = new PostgresOrderRepo(testDb);
  });

  it('persists and retrieves order', async () => {
    const order = Order.create({ ... });
    await repo.save(order);

    const found = await repo.findById(order.id);
    expect(found).toEqual(order);
  });
});
```

---

## High-Value Integration Tests

Focus integration tests on:

1. **Boundaries** - Where systems meet
2. **Critical paths** - Money, security, core features
3. **Complex queries** - Database operations

### Contract Tests

Verify implementations match interfaces.

```typescript
// Shared contract test
function testUserRepoContract(createRepo: () => UserRepo) {
  describe('UserRepo Contract', () => {
    let repo: UserRepo;

    beforeEach(() => {
      repo = createRepo();
    });

    it('saves and retrieves user', async () => {
      const user = User.create({ name: 'Test' });
      await repo.save(user);
      const found = await repo.findById(user.id);
      expect(found).toEqual(user);
    });

    it('returns null for missing user', async () => {
      const found = await repo.findById('nonexistent');
      expect(found).toBeNull();
    });
  });
}

// Apply to all implementations
testUserRepoContract(() => new InMemoryUserRepo());
testUserRepoContract(() => new PostgresUserRepo(testDb));
```

---

## Test Builders

Create test objects easily.

```typescript
class OrderBuilder {
  private props: Partial<OrderProps> = {
    id: 'order-1',
    customerId: 'cust-1',
    items: [],
    status: 'pending',
  };

  withId(id: string): OrderBuilder {
    this.props.id = id;
    return this;
  }

  withItems(items: Item[]): OrderBuilder {
    this.props.items = items;
    return this;
  }

  paid(): OrderBuilder {
    this.props.status = 'paid';
    return this;
  }

  build(): Order {
    return Order.create(this.props as OrderProps);
  }
}

// Usage
const order = new OrderBuilder()
  .withItems([{ sku: 'ABC', price: 100 }])
  .paid()
  .build();
```

---

## Common Testing Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Testing implementation | Brittle tests | Test behavior only |
| Too many mocks | Tests prove nothing | Use real objects when possible |
| Shared state | Flaky tests | Isolate each test |
| No assertions | False confidence | Always assert something meaningful |
| Testing trivial code | Wasted effort | Focus on logic and edge cases |
| Slow tests | Reduced feedback | Optimize, use unit tests |
